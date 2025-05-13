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

parser.add_argument('--mode', type=str, default='random')
parser.add_argument('--generate_num', type=int, default=10)
parser.add_argument('--sample_range', type=float, default=0.1)

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
input_dataloader = DataLoader(dataset=input_dataset, batch_size=1, shuffle = False)

z_list = []
for idx, data in enumerate(input_dataloader):
    logp, mu, logv, z, prediction = model(to_var(data['input'], gpu_exist = True),to_var(data['length'], gpu_exist = True))
    z_list.append(z)

z_tensor = torch.cat(z_list, dim=0).view(-1, args.latent_size)
smiles_encode = z_tensor.to('cpu').numpy()

smiles_list = []
prop_list = []
if args.mode == "random":
    z_new = torch.randn(args.generate_num, args.latent_size).cuda()
    for z_idx in range(z_new.size(0)):
        with torch.no_grad():
            out = model.inference(z_new[z_idx].unsqueeze(0))
            reconst = [dm_prop.ind2char[j] for j in out]
            if reconst[-1] == '[E]':
                reconst = reconst[:-1]
            selfies = ''.join(reconst)
            smiles = sf.decoder(selfies)
            smiles_list.append(smiles)
            prop_list.append(model.predictor(z_new[z_idx].unsqueeze(0)).cpu())
elif args.mode == "specify":
    z_new = z_tensor
    z_new_list = [z_new]
    for i in range(args.generate_num):
        Rand_sample = to_var(torch.normal(mean=torch.zeros([z.size(0), args.latent_size]), std=args.sample_range * torch.ones([z.size(0), args.latent_size])), gpu_exist=True)
        z_new_list.append(z_new + Rand_sample)
    for step in range(len(z_new_list)):
        for z_idx in range(z_new_list[step].size(0)):
            with torch.no_grad():
                out = model.inference(z_new_list[step][z_idx].unsqueeze(0))
                reconst = [dm_prop.ind2char[j] for j in out]
                if reconst[-1] == '[E]':
                    reconst = reconst[:-1]
                selfies = ''.join(reconst)
                smiles = sf.decoder(selfies)
                smiles_list.append(smiles)
                prop_list.append(model.predictor(z_new_list[step][z_idx].unsqueeze(0)).cpu())

prop_list = torch.cat(prop_list, dim=0)
HLGap_list = prop_list[:,0]
IP_list = prop_list[:,1]
EA_list = prop_list[:,2]
Dipole_list = prop_list[:,3]
OpticalGap_list = prop_list[:,4]
if args.mode == "random":
    results_df = pd.DataFrame({'Sample_No': [i for i in range(args.generate_num)], 'SMILES': smiles_list, 'HLGap': HLGap_list, 'IP': IP_list, 'EA': EA_list, 'Dipole': Dipole_list, 'OpticalGap': OpticalGap_list})
elif args.mode == "specify":
    results_df = pd.DataFrame({'Sample_No': [i for i in range(args.generate_num + 1) for _ in range(10)], 'idx':[idx for _ in range(args.generate_num + 1) for idx in range(10)], 'SMILES': smiles_list, 'HLGap': HLGap_list, 'IP': IP_list, 'EA': EA_list, 'Dipole': Dipole_list, 'OpticalGap': OpticalGap_list})

output_dir = "../result/sample/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
results_df.to_csv(output_dir + f"{args.mode}_sample.csv", index=False)
