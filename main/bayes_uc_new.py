#!/usr/bin/env python
# ==============================================================
#  bayes_uc.py  (uncertainty‑aware version, 2025‑05‑08)
#  原脚本作者：Loo Ting
#  维护：ChatGPT   —   仅在原版本基础上插入属性预测器
#                      MC‑Dropout 估计与 BO 联合不确定性
# ==============================================================

import os, sys, random, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import qNoisyExpectedImprovement      # ==== MOD ====

import py3Dmol
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Fragments
RDLogger.DisableLog("rdApp.*")

# ------------------------------------------------------------------
#  项目内部依赖
# ------------------------------------------------------------------
sys.path.append(os.path.join(os.getcwd(), ".."))
from src.model_cov import Model, to_var
from src.data_prepration import dataset_building
from src.util import text2dict_zinc, smi_postprocessing, decode
from src.les_penalty import LESModelWrapper, LES
# ==== MOD BEGIN ====================================================
from src.uncertainty_utils import PredictorUncertaintyWrapper
# ==== MOD END ======================================================

# ---------------- 全局常量（可自行调整） ---------------------------
MC_SAMPLES   = 20     # predictor 前向随机采样次数 (MC‑Dropout)
BETA_PENALTY = 2.0    # 综合得分中  β·σ  惩罚系数
UNC_THR      = 5.0    # GP 采样点的允许 σ 上限（原逻辑保留）
TARGET_PROP  = 'OpticalGap'
# ------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.double

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str,
                    default="../result/optimization/bo",
                    help="保存目录")
parser.add_argument("--seed",  type=int, default=42)
args, _ = parser.parse_known_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.outdir, exist_ok=True)

# ====================== 1. 载入模型与数据 ==========================
dm_prop   = pickle.load(open('../data/processed/dm_prop.pkl', 'rb'))
vocab_size, sos_idx, eos_idx, pad_idx = dm_prop.vocab_size, dm_prop.sos_indx, \
                                        dm_prop.eos_indx, dm_prop.pad_indx
char2ind, max_sequence_length = dm_prop.char2ind, dm_prop.max_length
prop_idx  = {'HLGap':0, 'IP':1, 'EA':2, 'OpticalGap':3}

NLL = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_idx)

print("Loading VAE …")
model = Model(
    vocab_size=vocab_size, embedding_size=30, rnn_type='gru', hidden_size=1024,
    word_dropout=0.0, latent_size=192, sos_idx=sos_idx, eos_idx=eos_idx,
    pad_idx=pad_idx, max_sequence_length=240, num_layers=1,
    prop_hidden_size=[256,192,128,64,32],
    predict_prop=True, bidirectional=False, gpu_exist=True, NLL=NLL
).to(device)
model.load_state_dict(torch.load('../checkpoints/vae_prop/checkpoint.pt',
                                 map_location=device))
model.eval()

# ==== MOD BEGIN: 包装 predictor 获取均值+方差 =======================
uc_wrapper = PredictorUncertaintyWrapper(
    model.predictor, mode="mc_dropout", n_samples=MC_SAMPLES
).to(device)
# ==== MOD END ======================================================

#  LES 惩罚模块（原逻辑）
les_wrapper       = LESModelWrapper(model)
les_penalty_model = LES(les_wrapper).to(device)

# ====================== 2. 实用函数 ================================
def is_imide_monomer(mol_or_smiles) -> bool:
    """判定是否酰亚胺单体（原函数保留，略）"""
    IMIDE_SMARTS = '[CX3H0](=O)(*)N[CX3H0](=O)*'
    PATTERN      = Chem.MolFromSmarts(IMIDE_SMARTS)
    if isinstance(mol_or_smiles, Chem.Mol):
        mol = mol_or_smiles
    else:
        mol = Chem.MolFromSmiles(str(mol_or_smiles))
        if mol is None: return False
    if not mol.GetSubstructMatches(PATTERN, uniquify=True): return False
    if Fragments.fr_imide(mol) == 0: return False
    return True

def compute_diversity_penalty(smiles, thr=0.3, p=10.0):
    tokens = list(smiles)
    if not tokens: return 0.0
    r = len(set(tokens)) / len(tokens)
    return p * (thr - r) if r < thr else 0.0

# ------------------------------------------------------------------
#  新版 evaluate_candidate: 计算 μ, σ, penalty
# ------------------------------------------------------------------
def evaluate_candidate(z_latent: torch.Tensor):
    """
    输入  z_latent:(1,D cuda) → (score, details)
    details 字段含新增 opt_gap_mu/opt_gap_std
    """
    with torch.no_grad():
        # 解码 SELFIES -> SMILES
        seq  = model.inference(z_latent.float())
        chars = [dm_prop.ind2char[i] for i in seq]
        if chars[-1] == '[E]': chars = chars[:-1]
        selfies_str = ''.join(chars)
        smiles      = sf.decoder(selfies_str)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -10.0, {'smiles':smiles, 'valid':False}

    # =========== 属性预测 + 不确定性 (μ, σ) ========================
    with torch.no_grad():
        mu_pred, std_pred = uc_wrapper(z_latent)
    mu_pred  = mu_pred.squeeze(0).to(device)
    std_pred = std_pred.squeeze(0).to(device)

    # 目标属性值与不确定度
    opt_mu  = float(mu_pred[prop_idx[TARGET_PROP]])
    opt_std = float(std_pred[prop_idx[TARGET_PROP]])

    # 其余属性（如 HLGap/IP/EA）若需要同理获取
    property_values = {TARGET_PROP: opt_mu}

    # SA_score
    try:
        sa_score = float(sascorer.calculateScore(mol))
    except Exception:
        sa_score = 10.0
    property_values['SA_score'] = sa_score

    # 酰亚胺判定
    not_imide_penalty = 0.0 if is_imide_monomer(mol) else 10.0

    # 多样性惩罚
    div_pen = compute_diversity_penalty(smiles, thr=0.3, p=10.0)

    # LES penalty
    with torch.no_grad():
        les_val = float(les_penalty_model(z_latent).item())

    # =========  综合得分  =========================================
    #   maximize OpticalGap, penalise uncertainty β·σ  + 其他惩罚
    composite = opt_mu - BETA_PENALTY*opt_std - sa_score*0.3 \
                - not_imide_penalty - div_pen

    details = dict(smiles=smiles, valid=True, SA_score=sa_score,
                   properties=property_values,
                   opt_gap_mu=opt_mu, opt_gap_std=opt_std,
                   composite_score=composite,
                   diversity_penalty=div_pen,
                   les_penalty=les_val,
                   is_imide=not_imide_penalty==0,
                   latent_vector=z_latent.detach().cpu().numpy().squeeze().tolist())
    return composite, opt_std**2 + 1e-6, details      # 返回 noise 供 GP

# ====================== 3. 初始数据集 =============================
input_csv = '../data/raw/BO_initial_data_test.csv'
raw_dict  = smi_postprocessing(text2dict_zinc(input_csv), max_length=max_sequence_length)
dataset   = dataset_building(char2ind, raw_dict, max_sequence_length, 'pure_smiles')
loader    = DataLoader(dataset, batch_size=1, shuffle=False)

X0, Y0, Yvar0, history = [], [], [], []
for i, batch in enumerate(loader):
    _,_,_, z, _ = model(to_var(batch['input'], gpu_exist=True),
                        to_var(batch['length'], gpu_exist=True))
    z = z.to(device)
    s, n, det = evaluate_candidate(z)
    X0.append(z.squeeze(0)).to(device)
    Y0.append([s])
    Yvar0.append([n])
    det.update(iteration=0, index=i);  history.append(det)

train_X   = torch.stack(X0).double(); train_Y  = torch.tensor(Y0).double()
train_Yvar= torch.tensor(Yvar0).double()

d_latent  = train_X.size(-1)
bounds = torch.stack([torch.full((d_latent,), -0.35),
                      torch.full((d_latent,),  0.35)]).double()
bounds = bounds.to(device)

# ====================== 4. 贝叶斯优化循环 ==========================
BO_ITER, BATCH_Q = 5, 1
for it in tqdm(range(1, BO_ITER+1), desc="BayesOpt"):
    gp = SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar)
    gp = gp.to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll.train(); gp.train()
    torch.optim.Adam(gp.parameters(), lr=0.1).zero_grad()
    mll(gp(train_X), train_Y.squeeze(-1)).backward()
    torch.optim.Adam(gp.parameters(), lr=0.1).step()
    gp.eval()

    acqf = qNoisyExpectedImprovement(model=gp, X_baseline=train_X)
    cand, _ = optimize_acqf(acqf, bounds=bounds, q=BATCH_Q,
                            num_restarts=8, raw_samples=32)

    # 评估并追加
    for j in range(cand.size(0)):
        z_new = cand[j:j+1].to(device)
        s, n, det = evaluate_candidate(z_new)
        train_X   = torch.cat([train_X, z_new.to(device).double()], dim=0)
        train_Y   = torch.cat([train_Y, torch.tensor([[s]]).double()], dim=0)
        train_Yvar= torch.cat([train_Yvar,
                               torch.tensor([[n]]).double()], dim=0)
        det.update(iteration=it, batch_index=j,
                   global_index=len(history))
        history.append(det)

# ====================== 5. 结果保存与可视化 =========================
hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(args.outdir,"bo_all_candidates.csv"), index=False)

#  PCA 不确定度热图
lat_mat = np.vstack(hist_df['latent_vector'])
std_arr = hist_df['opt_gap_std'].to_numpy()
proj    = PCA(n_components=2).fit_transform(lat_mat)
plt.figure(figsize=(5,4))
sc=plt.scatter(proj[:,0], proj[:,1], c=std_arr, s=15, cmap='viridis')
plt.colorbar(sc,label='Predictor σ (OpticalGap)')
plt.title('Latent‑space uncertainty (PCA)')
plt.tight_layout()
plt.savefig(os.path.join(args.outdir,"uncertainty_pca.png"), dpi=300)
print("✓  BO 结束，结果已写入", args.outdir)
