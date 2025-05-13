import pickle
import os
import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

from src.util import text2dict_zinc, smi_postprocessing
from src.data_prepration import dataset_building
from src.model import Model, to_var
from src.util import decode
import pandas as pd
import selfies as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=30)
parser.add_argument('--rnn_type', type=str, default='gru')
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--word_dropout', type=float, default=0.0)
parser.add_argument('--latent_size', type=int, default=196)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--prop_hidden_size', type=list, default=[256, 192, 128, 64, 32])

parser.add_argument('--model_path', type=str, default='../checkpoints/vae_prop/checkpoint.pt')
parser.add_argument('--input_path', type=str, default='../data/raw/PI_100K_test.csv')
parser.add_argument('--output_path', type=str, default='../result/visualization/')

parser.add_argument('--property', type=str, default='HLGap')
parser.add_argument('--batch_size', type=int, default=256)

if os.path.exists('../data/processed/dm_prop.pkl'):
    dm_prop = pickle.load(open('../data/processed/dm_prop.pkl', 'rb'))

args = parser.parse_args()
vocab_size = dm_prop.vocab_size
sos_idx = dm_prop.sos_indx
eos_idx = dm_prop.eos_indx
pad_idx = dm_prop.pad_indx
max_sequence_length = dm_prop.max_length
char2ind = dm_prop.char2ind
prop_idx = {'HLGap':0, 'IP':1, 'EA':2, 'Dipole':3, 'OpticalGap':4}

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

NLL = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_idx)

model = Model(vocab_size=vocab_size, embedding_size=args.embedding_size, rnn_type=args.rnn_type, hidden_size=args.hidden_size, word_dropout=args.word_dropout, latent_size=args.latent_size,
                sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx, max_sequence_length=max_sequence_length, num_layers=args.num_layers, prop_hidden_size=args.prop_hidden_size, 
                predict_prop = True, bidirectional=False, gpu_exist = True, NLL=NLL)

print('Loading model ...')
model.load_state_dict(torch.load(args.model_path))
for p in model.parameters():
    p.requires_grad = False
print('Model: ', model)
model.eval()
model.cuda()

prop_data = pd.read_csv(args.input_path)
prop_data = prop_data[prop_data['smiles'].str.len() < 240]

input_data = text2dict_zinc(args.input_path)
input_data_ = smi_postprocessing(input_data, max_length=240)

input_dataset = dataset_building(char2ind,input_data_,dm_prop.max_length,'pure_smiles')
input_dataloader = DataLoader(dataset=input_dataset, batch_size=1, shuffle=False, drop_last=False)

z_list = []
for idx, data in enumerate(input_dataloader):
    logp, mu, logv, z, prediction = model(to_var(data['input'], gpu_exist = True),to_var(data['length'], gpu_exist = True))
    z_list.append(z)

z_tensor = torch.cat(z_list, dim=0).view(-1, 196)
pred_prop = model.predictor(z_tensor)

properties = ['HLGap', 'IP', 'EA', 'Dipole', 'OpticalGap']
metrics = {'mae': mean_absolute_error, 'mse': root_mean_squared_error, 'rmse': lambda y_true, y_pred: np.sqrt(root_mean_squared_error(y_true, y_pred)), 'r2': r2_score}

result = []
for i in range(5):
    result.append(list(metrics[metric](prop_data[properties[i]].to_numpy(), pred_prop[:, i].cpu().detach().numpy()) for metric in metrics))

result_array = np.array(result)
results_df = pd.DataFrame({'property': properties, 'MAE': result_array[:, 0], 'MSE': result_array[:, 1], 'RMSE': result_array[:, 2], 'R2': result_array[:, 3]})

output_dir = "../result/evaluation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
results_df.to_csv(f"../result/evaluation/result.csv", index=False)
