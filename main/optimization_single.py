import pickle
import os
import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim

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
parser.add_argument('--input_path', type=str, default='../data/raw/PI_100K_generation.csv')

parser.add_argument('--property', type=str, default='HLGap')
parser.add_argument('--maximize', type=bool, default=True)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--num_steps', type=int, default=10)

if os.path.exists('../data/processed/dm_prop.pkl'):
    dm_prop = pickle.load(open('../data/processed/dm_prop.pkl', 'rb'))

args = parser.parse_args()
vocab_size = dm_prop.vocab_size
sos_idx = dm_prop.sos_indx
eos_idx = dm_prop.eos_indx
pad_idx = dm_prop.pad_indx
max_sequence_length = dm_prop.max_length
char2ind = dm_prop.char2ind

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

input_data = text2dict_zinc(args.input_path)
input_data_ = smi_postprocessing(input_data, max_length=max_sequence_length)

input_dataset = dataset_building(char2ind,input_data_,dm_prop.max_length,'pure_smiles')
input_dataloader = DataLoader(dataset=input_dataset, batch_size= 1, shuffle = False)

z_list = []
for idx, data in enumerate(input_dataloader):
    logp, mu, logv, z, prediction = model(to_var(data['input'], gpu_exist = True),to_var(data['length'], gpu_exist = True))
    z_list.append(z)

num_molecules=len(z_list)
z_tensor = torch.cat(z_list, dim=0).view(len(z_list), args.latent_size)

def get_optimized_z_single(z_input, model, property, maximize, step_size=0.1, num_steps=50):
    prop_idx = {'HLGap':0, 'IP':1, 'EA':2, 'Dipole':3, 'OpticalGap':4}
    z0 = z_input.clone().detach()
    z0.requires_grad = True
    losses = []
    z_processed = []
    prop = []
    optimizer = optim.Adam([z0], lr=step_size)
    for epoch in tqdm(range(num_steps+1), desc='generating molecules'):
        loss = 0
        optimizer.zero_grad()
        out = model.predictor(z0)[:, prop_idx[property]]
        if maximize:
            loss -= torch.mean(out)
        else:
            loss += torch.mean(out)
        z_processed.append(z0.clone())
        prop.append(out)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return z_processed, prop

z_optimized, prop_optimized = get_optimized_z_single(z_input=z_tensor, model=model, property=args.property, maximize=args.maximize, step_size=0.01, num_steps=10)

smiles_list = []
prop_list = []
for i in range(len(z_optimized)):
    for z_idx in range(z_optimized[i].size(0)):
        with torch.no_grad():
            out = model.inference(z_optimized[i][z_idx].unsqueeze(0))
            reconst = [dm_prop.ind2char[j] for j in out]
            if reconst[-1] == '[E]':
                reconst = reconst[:-1]
            selfies = ''.join(reconst)
            smiles = sf.decoder(selfies)
            smiles_list.append(smiles)
            prop_list.append(prop_optimized[i][z_idx])

results_df = pd.DataFrame({'Steps': [i for i in range(0, 11) for _ in range(num_molecules)], 'idx':[idx for _ in range(0, 11) for idx in range(num_molecules)], 'SMILES': smiles_list, f'{args.property}': [prop.item() for prop in prop_list]})
output_dir = "../result/optimization/single"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if args.maximize:
    results_df.to_csv(f"../result/optimization/single/{args.property}_maximize.csv", index=False)
else:
    results_df.to_csv(f"../result/optimization/single/{args.property}_minimize.csv", index=False)