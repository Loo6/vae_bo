import os
import sys
# 获取当前代码文件绝对路径
current_dir = os.getcwd()
# 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到 sys.path 中
sys.path.append(os.path.join(current_dir, ".."))
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import pandas as pd

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import qExpectedImprovement
from botorch import fit_gpytorch_mll

import py3Dmol
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles, RDConfig
# 禁用 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# 将 RDKit 自带的 SA_Score 模块加入路径并导入
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from src.util import text2dict_zinc, smi_postprocessing
from src.data_prepration import dataset_building
from src.model_cov import Model, to_var  # 假设使用原分子VAE模型结构
from src.util import decode as decode_molecule  # 原分子的解码函数

from torch.utils.data import DataLoader
import torch.nn as nn

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# 文件路径设置
model_path = '../checkpoints/vae_prop/checkpoint.pt'
input_path = '../data/raw/BO_initial_data.csv'
output_path = '../result/optimization/bo'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 加载预处理信息
dm_prop = pickle.load(open('../data/processed/dm_prop.pkl', 'rb'))
vocab_size = dm_prop.vocab_size
sos_idx = dm_prop.sos_indx
eos_idx = dm_prop.eos_indx
pad_idx = dm_prop.pad_indx
max_sequence_length = dm_prop.max_length
char2ind = dm_prop.char2ind

# 定义属性索引映射（原预测器支持的属性）
prop_idx = {'HLGap': 0, 'IP': 1, 'EA': 2, 'OpticalGap': 3}

# 定义目标属性及优化配置（注意：SA_score 越低越好，此处暂未使用 SA_score 项）
target_properties = {
    # 'HLGap': {'opt_direction': 'max', 'weight': 1.0, 'range': (6.0, 7.0)},
    'OpticalGap': {'opt_direction': 'max', 'weight': 1.0, 'range': (5.0, 7.0)},
    # 'SA_score': {'opt_direction': 'min', 'weight': 1.0, 'range': (1.0, 10.0)}
}

# 无效分子惩罚得分（目标为最大化时使用）
INVALID_SCORE = -10.0

# 定义交叉熵损失（训练时使用，本处仅作传参）
NLL = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_idx)

# 加载 VAE 模型（包含属性预测器）
print('Loading model ...')
model = Model(
    vocab_size=vocab_size, embedding_size=30, rnn_type='gru', hidden_size=1024, word_dropout=0.0, latent_size=192,
    sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx, max_sequence_length=240, num_layers=1,
    prop_hidden_size=[256, 192, 128, 64, 32],
    predict_prop=True, bidirectional=False, gpu_exist=True, NLL=NLL
)
model.load_state_dict(torch.load(model_path))
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.cuda()

###########################################
# 检查是否为酰亚胺单体的函数
###########################################
def is_imide_monomer(mol):
    """
    判断分子是否为酰亚胺单体。
    这里采用一个简单的 SMARTS 模式（例如 succinimide 模式），请根据实际需求调整。
    """
    imide_smarts = "O=C1NC(=O)C1"  # 简化版酰亚胺结构
    patt = Chem.MolFromSmarts(imide_smarts)
    return mol.HasSubstructMatch(patt)

###########################################
# 定义额外的惩罚函数：多样性（避免重复单元）
###########################################
def compute_diversity_penalty(smiles, threshold=0.3, penalty_value=10.0):
    """
    根据 SMILES 中唯一字符占比进行简单多样性评价。
    如果比例低于 threshold，则给予惩罚。
    """
    tokens = list(smiles)
    if len(tokens) == 0:
        return 0.0
    ratio = len(set(tokens)) / len(tokens)
    if ratio < threshold:
        return penalty_value * (threshold - ratio)
    else:
        return 0.0

###########################################
# 评估函数：解码、有效性检查、属性计算与惩罚整合
###########################################
def evaluate_candidate(latent_vector):
    """
    对给定 latent_vector（形状 (1, latent_dim)）进行解码、检查、属性计算，并返回综合目标得分及详细信息。
    """
    # 解码：利用 model.inference 得到字符序列，再转换为 SELFIES，再解码为 SMILES
    with torch.no_grad():
        out = model.inference(latent_vector.cuda().float())
    decoded_seq = [dm_prop.ind2char[j] for j in out]
    if decoded_seq[-1] == '[E]':
        decoded_seq = decoded_seq[:-1]
    selfies_str = ''.join(decoded_seq)
    smiles = sf.decoder(selfies_str)  # 直接用 SELFIES 解码保证鲁棒性
    
    # 检查 SMILES 有效性
    mol = MolFromSmiles(smiles)
    if mol is None:
        details = {
            'smiles': smiles,
            'valid': False,
            'SA_score': None,
            'properties': {},
            'latent_vector': latent_vector.detach().cpu().numpy().squeeze().tolist(),
            'composite_score': INVALID_SCORE,
            'not_imide': None
        }
        return INVALID_SCORE, details

    # 判断是否为酰亚胺单体，若不是，施加额外惩罚
    if not is_imide_monomer(mol):
        not_imide_penalty = 10.0
        is_imide = False
    else:
        not_imide_penalty = 0.0
        is_imide = True

    # 计算 SA_score（如果需要，可以加入该项，目前代码中未启用）
    try:
        sa_score = sascorer.calculateScore(mol)
    except Exception as e:
        sa_score = 10.0

    # 计算其他属性（利用预测器）
    property_values = {}
    with torch.no_grad():
        pred_vals = model.predictor(latent_vector.cuda().float())
    for prop, config in target_properties.items():
        if prop == 'SA_score':
            property_values[prop] = sa_score
        else:
            if prop in prop_idx:
                property_values[prop] = pred_vals[0, prop_idx[prop]].item()
            else:
                property_values[prop] = 0.0

    # 检查各属性是否满足允许范围，不满足则施加硬惩罚
    penalty = 0.0
    for prop, config in target_properties.items():
        if 'range' in config:
            min_val, max_val = config['range']
            val = property_values[prop]
            if val is None:
                penalty += 10.0
            else:
                if val < min_val or val > max_val:
                    penalty += 5.0

    # 计算初步综合目标得分：对于“max”直接加；对于“min”取负（转化为最大化问题）
    composite_score = 0.0
    for prop, config in target_properties.items():
        weight = config.get('weight', 1.0)
        direction = config.get('opt_direction', 'max')
        val = property_values[prop]
        component = weight * (val if direction == 'max' else -val)
        composite_score += component
    composite_score -= penalty
    composite_score -= not_imide_penalty  # 加入酰亚胺判断惩罚

    # 计算多样性惩罚（避免重复结构）
    diversity_penalty = compute_diversity_penalty(smiles, threshold=0.3, penalty_value=10.0)
    composite_score -= diversity_penalty

    details = {
        'smiles': smiles,
        'valid': True,
        'SA_score': sa_score,
        'properties': property_values,
        'latent_vector': latent_vector.detach().cpu().numpy().squeeze().tolist(),
        'composite_score': composite_score,
        'diversity_penalty': diversity_penalty,
        'not_imide': not_imide_penalty > 0.0  # True 表示不符合酰亚胺单体要求
    }
    return composite_score, details

###########################################
# 分子结构三维可视化函数，添加保存图像选项
###########################################
def visualize_molecule_3d(smiles, width=500, height=500, save_path=None, show_view=False):
    """
    利用 RDKit 生成 3D 坐标，并使用 py3Dmol 可视化分子结构。
    当提供 save_path 时，尝试保存图像：
      - 首先尝试保存 PNG 图像（需要 Notebook 环境），
      - 如果失败则尝试保存 HTML 文件（利用 view.getView()）。
    参数 show_view 为 True 时尝试显示交互视图，否则返回 view 对象。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid molecule:", smiles)
        return
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if status < 0 or mol.GetNumConformers() == 0:
        print("Embedding failed for molecule:", smiles)
        return
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        print("UFF optimization failed for molecule:", smiles)
        return
    mol_block = Chem.MolToMolBlock(mol)
    
    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_block, 'mol')
    view.setStyle({'stick': {}})
    view.zoomTo()
    view.render()
    
    if save_path is not None:
        # 首先尝试保存为 PNG
        try:
            png_str = view.png()  # 尝试获取 PNG 图片（base64编码）
            import base64
            with open(save_path, "wb") as f:
                f.write(base64.b64decode(png_str))
            print("Saved 3D visualization as PNG:", save_path)
        except Exception as e:
            print("Saving 3D visualization as PNG failed:", e)
            # 尝试保存为 HTML
            try:
                # 尝试调用 getView() 获取 HTML 字符串
                html_str = view.getView()
                if not isinstance(html_str, str):
                    html_str = str(html_str)
                html_save_path = save_path.replace(".png", ".html")
                with open(html_save_path, "w", encoding="utf-8") as f:
                    f.write(html_str)
                print("Saved 3D visualization as HTML:", html_save_path)
            except Exception as e2:
                print("Saving 3D visualization as HTML failed:", e2)
    
    try:
        if show_view:
            return view.show()
        else:
            return view
    except Exception as e:
        print("Interactive view not available:", e)
        return view


###########################################
# 可视化代表候选分子（3D），打印SMILES及属性信息
###########################################
def visualize_representative_candidates(representatives, max_display=10, save_dir=None):
    count = 0
    for rep in representatives:
        if rep.get('valid', False):
            smiles = rep.get('smiles', '')
            iter_info = rep.get('iteration', '?')
            batch_info = rep.get('batch_index', '?')
            global_idx = rep.get('global_index', '?')
            properties = rep.get('properties', {})
            sa_score = rep.get('SA_score', '')
            print(f"GlobalIdx: {global_idx} | Iteration: {iter_info} | Batch: {batch_info} | SMILES: {smiles}")
            print(f"Properties: {properties} | SA_score: {sa_score}")
            # 若提供保存目录，则构造保存路径（文件名含迭代信息）
            save_path = None
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"rep_iter{iter_info}_batch{batch_info}_global{global_idx}.png")
            visualize_molecule_3d(smiles, width=500, height=500, save_path=save_path)
            count += 1
            if count >= max_display:
                break

###########################################
# 初始训练数据构造：优质+随机混合（这里仅取优质样本）
###########################################
input_data = text2dict_zinc(input_path)
input_data_processed = smi_postprocessing(input_data, max_length=max_sequence_length)
input_dataset = dataset_building(char2ind, input_data_processed, dm_prop.max_length, 'pure_smiles')
input_dataloader = DataLoader(dataset=input_dataset, batch_size=1, shuffle=False)

z_list = []
pred_scores = []
for idx, data in enumerate(input_dataloader):
    logp, mu, logv, z, prediction = model(to_var(data['input'], gpu_exist=True), to_var(data['length'], gpu_exist=True))
    z_list.append(z)
    pred_scores.append(prediction[:, prop_idx['OpticalGap']].item())
num_molecules = len(z_list)
latent_dim = z_list[0].shape[-1]
z_tensor = torch.cat(z_list, dim=0).view(num_molecules, latent_dim)

elite_fraction = 0.1
num_elite = int(elite_fraction * num_molecules)
sorted_indices = np.argsort(pred_scores)[::-1]
elite_indices = sorted_indices[:num_elite]
init_indices = np.unique(np.concatenate([elite_indices]))
initial_x = z_tensor[init_indices].detach().cpu().double()
initial_y_list = []
eval_history = []
for i in range(initial_x.size(0)):
    latent = initial_x[i].unsqueeze(0)
    score, details = evaluate_candidate(latent)
    initial_y_list.append(score)
    details['iteration'] = 0
    details['index'] = i
    eval_history.append(details)
initial_y = torch.tensor(initial_y_list).unsqueeze(-1).double()

# 保存初始数据到 CSV（可选）
initial_df = pd.DataFrame(eval_history)
initial_df.to_csv(os.path.join(output_path, "initial_candidates.csv"), index=False)

d = latent_dim
bounds = torch.stack([torch.full((d,), -0.3), torch.full((d,), 1)]).double()

###########################################
# BO 优化循环（自适应调整候选数量）
###########################################
bo_iterations = 100
batch_size = 10
max_batch_size = 50

all_x = initial_x.clone()
all_y = initial_y.clone()
bo_x_list = []
bo_y_list = []
acq_values = []
iteration_representatives = []  # 记录每轮代表候选

no_improve_count = 0
improve_threshold = 1e-3
best_so_far = all_y.max().item()

for bo_iter in tqdm(range(bo_iterations), desc='Bayesian Optimization'):
    gp = SingleTaskGP(all_x, all_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    best_f = all_y.max().item()
    EI = qExpectedImprovement(model=gp, best_f=best_f)
    
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds, q=batch_size, num_restarts=10, raw_samples=20
    )
    acq_values.append(acq_value.item())
    
    candidate = candidate.detach().cpu().double()
    current_iter_details = []

    # -------- 获取 GP 对候选点的均值和方差 --------
    with torch.no_grad():
        posterior = gp.posterior(candidate)
        mean = posterior.mean.squeeze(-1).cpu().numpy()
        std = posterior.variance.sqrt().squeeze(-1).cpu().numpy()
    # ------------------------------------------

    # 设置不确定性阈值（如0.08，可根据实际调整）
    uncertainty_threshold = 2

    for i in range(candidate.shape[0]):
        # 跳过不确定性过大的候选
        if std[i] > uncertainty_threshold:
            continue
        cand = candidate[i].unsqueeze(0)
        score, details = evaluate_candidate(cand)
        details['iteration'] = bo_iter + 1
        details['batch_index'] = i
        details['global_index'] = len(eval_history)
        # 记录不确定性信息
        details['gp_mean'] = float(mean[i])
        details['gp_std'] = float(std[i])
        eval_history.append(details)
        current_iter_details.append(details)
        
        score_tensor = torch.tensor([score]).unsqueeze(-1).double()
        all_x = torch.cat([all_x, cand], dim=0)
        all_y = torch.cat([all_y, score_tensor], dim=0)
        bo_x_list.append(cand)
        bo_y_list.append(score)
    
    # 若本轮所有候选都被跳过，则放宽阈值或强制采集最小不确定性者
    if not current_iter_details and candidate.shape[0] > 0:
        min_idx = int(np.argmin(std))
        cand = candidate[min_idx].unsqueeze(0)
        score, details = evaluate_candidate(cand)
        details['iteration'] = bo_iter + 1
        details['batch_index'] = min_idx
        details['global_index'] = len(eval_history)
        details['gp_mean'] = float(mean[min_idx])
        details['gp_std'] = float(std[min_idx])
        eval_history.append(details)
        current_iter_details.append(details)
        score_tensor = torch.tensor([score]).unsqueeze(-1).double()
        all_x = torch.cat([all_x, cand], dim=0)
        all_y = torch.cat([all_y, score_tensor], dim=0)
        bo_x_list.append(cand)
        bo_y_list.append(score)

    if current_iter_details:
        # 代表分子选择时也可避开高不确定性
        rep = max(current_iter_details, key=lambda d: d['composite_score'])
        iteration_representatives.append(rep)
    
    current_best = all_y.max().item()
    if current_best - best_so_far < improve_threshold:
        no_improve_count += 1
    else:
        no_improve_count = 0
        best_so_far = current_best

    if no_improve_count >= 3 and batch_size < max_batch_size:
        batch_size += 1
        no_improve_count = 0
        print(f"Iteration {bo_iter}: Increasing batch size to {batch_size} due to slow improvement.")

###########################################
# 保存所有 BO 过程中产生的候选分子信息到 CSV
###########################################
df = pd.DataFrame(eval_history)
df.to_csv(os.path.join(output_path, "bo_all_candidates.csv"), index=False)

###########################################
# 找出最佳候选分子并显示结果
###########################################
if all_y.numel() > 0:
    best_idx = all_y.argmax().item()
    best_x = all_x[best_idx]
    best_score = all_y[best_idx].item()
    best_details = next((d_item for d_item in eval_history if abs(d_item['composite_score'] - best_score) < 1e-6), {})
else:
    best_score = None
    best_details = {}

print(f"Best composite score: {best_score}")
if best_details.get('valid', False):
    print(f"Best SMILES: {best_details.get('smiles', 'N/A')}")
else:
    print("Best candidate is an invalid molecule.")

###########################################
# 可视化
###########################################
# (a) 属性趋势图（折线图，不显示数据点标签，使用经典配色）
rep_iterations = [rep['iteration'] for rep in iteration_representatives]
rep_scores = [rep['composite_score'] for rep in iteration_representatives]

plt.figure()
plt.plot(rep_iterations, rep_scores, color='tab:blue', marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Composite Score')
plt.title('Optimization Progress (Representative per Iteration)')
plt.grid(True)
plt.savefig(os.path.join(output_path, "Optimization_Progress_(Representative_per_Iteration).png"), dpi=300)

# (a) 属性趋势图（折线图，不显示数据点标签，使用经典配色）
rep_iterations = [rep['iteration'] for rep in iteration_representatives]
rep_scores = [rep['properties']['OpticalGap'] for rep in iteration_representatives]

plt.figure()
plt.plot(rep_iterations, rep_scores, color='tab:blue', marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Optical Gap')
plt.title('Property Optimization Progress (Representative per Iteration)')
plt.grid(True)
plt.savefig(os.path.join(output_path, "Property_Optimization_Progress_(Representative_per_Iteration).png"), dpi=300)

# (b) 探索空间二维投影（不显示数据点标签，用迭代颜色映射）
rep_latent = np.array([rep['latent_vector'] for rep in iteration_representatives])
rep_iterations = [rep['iteration'] for rep in iteration_representatives]
pca = PCA(n_components=2)
proj = pca.fit_transform(rep_latent)
plt.figure()
sc = plt.scatter(proj[:, 0], proj[:, 1], c=rep_iterations, cmap='viridis', s=50)
plt.colorbar(sc, label='Iteration')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Representative Latent Space Trajectory (PCA Projection)')
plt.grid(True)
plt.savefig(os.path.join(output_path, "Representative_Latent_Space_Trajectory_(PCA_Projection).png"), dpi=300)

# (c) 合成性 VS 综合目标值（不显示数据点标签，用与二维投影相同的迭代颜色表示）
rep_sa_scores = [rep['SA_score'] if rep['SA_score'] is not None else 10.0 for rep in iteration_representatives]
rep_composite_scores = [rep['composite_score'] for rep in iteration_representatives]
plt.figure()
sc = plt.scatter(rep_sa_scores, rep_composite_scores, c=rep_iterations, cmap='viridis', s=50)
plt.colorbar(sc, label='Iteration')
plt.xlabel('SA_score (Synthetic Accessibility)')
plt.ylabel('Composite Score')
plt.title('SA_score vs Composite Score (Representative Candidates)')
plt.grid(True)
plt.savefig(os.path.join(output_path, "SA_score_vs_Composite_Score_(Representative_Candidates).png"), dpi=300)

# # (d) 三维结构可视化：展示代表候选分子，并保存图像到 output_path/3D_images/
# save_dir = os.path.join(output_path, "3D_images")
# visualize_representative_candidates(iteration_representatives, max_display=10, save_dir=save_dir)
