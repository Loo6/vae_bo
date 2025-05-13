import pickle
import os
import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.util import text2dict_zinc, smi_postprocessing
from src.data_prepration import dataset_building
from src.model_cov import Model, to_var
from src.util import decode
import pandas as pd
import selfies as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=30)
parser.add_argument('--rnn_type', type=str, default='gru')
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--word_dropout', type=float, default=0.0)
parser.add_argument('--latent_size', type=int, default=192)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--prop_hidden_size', type=list, default=[256, 192, 128, 64, 32])

parser.add_argument('--model_path', type=str, default='../checkpoints/vae_prop/checkpoint.pt')
parser.add_argument('--input_path', type=str, default='../data/raw/dft_predictions_latent_space.csv')
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

input_data = text2dict_zinc(args.input_path)
input_data_ = smi_postprocessing(input_data, max_length=max_sequence_length)

input_dataset = dataset_building(char2ind,input_data_,dm_prop.max_length,'pure_smiles')
input_dataloader = DataLoader(dataset=input_dataset, batch_size=args.batch_size, shuffle = False)

z_list = []
for idx, data in enumerate(input_dataloader):
    logp, mu, logv, z, prediction = model(to_var(data['input'], gpu_exist = True),to_var(data['length'], gpu_exist = True))
    z_list.append(z)

z_tensor = torch.cat(z_list, dim=0).view(-1, 192)
smiles_encode = z_tensor.to('cpu').numpy()

pred_prop = model.predictor(z_tensor)

pca = PCA(n_components=2)
pca.fit(smiles_encode)
z_pca = pca.transform(smiles_encode)

z_pca_x = z_pca[:, 0]
z_pca_y = z_pca[:, 1]

# Create Fig and gridspec
fig = plt.figure(figsize=(9, 8), dpi= 500)
grid = plt.GridSpec(2, 3, width_ratios=[1,0.31,0.2], height_ratios=[0.2,1], wspace=0.1, hspace=0.1) # Adjusted spacing

# Scatterplot on main ax
ax_main = fig.add_subplot(grid[1,:2])
scatter = ax_main.scatter(z_pca_x, z_pca_y, s=1, c=pred_prop[:, prop_idx[args.property]].cpu(), cmap="RdBu_r")
# Keep default ticks and spines, hide top and right spines
# ax_main.set_xticks([]) # Remove this line to show x-ticks
# ax_main.set_yticks([]) # Remove this line to show y-ticks
ax_main.spines['right'].set_visible(False)
ax_main.spines['top'].set_visible(False)
ax_main.tick_params(axis='x', labelbottom=True) # Ensure x-axis labels are shown
ax_main.tick_params(axis='y', labelleft=True) # Ensure y-axis labels are shown

# histogram on the up
ax_up = fig.add_subplot(grid[0,0], sharex=ax_main)
ax_up.hist(z_pca_x, 120, histtype='bar', orientation='vertical', color='#559FC9',edgecolor='k')
# ax_up.set_xticks([]) # Keep this to hide x-ticks on the top histogram
plt.setp(ax_up.get_xticklabels(), visible=False) # Hide x-tick labels
# ax_up.set_yticks([]) # Remove this line to show y-ticks
ax_up.spines['right'].set_visible(False)
ax_up.spines['top'].set_visible(False)
# ax_up.spines['left'].set_visible(False) # Remove this line to show left spine
ax_up.spines['bottom'].set_visible(False) # Hide bottom spine
ax_up.tick_params(axis='y', labelleft=True) # Ensure y-axis labels are shown

# histogram in the right
ax_right = fig.add_subplot(grid[1,2], sharey=ax_main)
ax_right.hist(z_pca_y, 120, histtype='bar', orientation='horizontal', color='#559FC9',edgecolor='k')
# ax_right.set_xticks([]) # Remove this line to show x-ticks
# ax_right.set_yticks([]) # Keep this to hide y-ticks on the right histogram
plt.setp(ax_right.get_yticklabels(), visible=False) # Hide y-tick labels
ax_right.spines['right'].set_visible(False)
ax_right.spines['top'].set_visible(False)
# ax_right.spines['bottom'].set_visible(False) # Remove this line to show bottom spine
ax_right.spines['left'].set_visible(False) # Hide left spine
ax_right.tick_params(axis='x', labelbottom=True) # Ensure x-axis labels are shown

# coloebar using the scatter plot object
cb1 = fig.colorbar(scatter, ax=ax_main, cmap="RdBu_r") # Use fig.colorbar for better placement control
cb1.set_label(f'{args.property}',labelpad=15,fontsize=14) # Adjusted labelpad

# Decorations
ax_main.set_xlabel('pca_x',fontsize=16,labelpad=8.5)
ax_main.set_ylabel('pca_y',fontsize=16,labelpad=11.5)
ax_up.set_ylabel('Frequency', fontsize=12) # Add label for top histogram y-axis
ax_right.set_xlabel('Frequency', fontsize=12) # Add label for right histogram x-axis

plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent title overlap if any
plt.savefig(args.output_path + f"LatentSpace_{args.property}.png")