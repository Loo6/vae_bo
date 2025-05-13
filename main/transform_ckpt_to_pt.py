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

checkpoint_path='../checkpoints/vae_prop/last.ckpt'
model_path = '../checkpoints/vae_prop/checkpoint.pt'

if os.path.exists('../data/processed/dm_prop.pkl'):
    dm_prop = pickle.load(open('../data/processed/dm_prop.pkl', 'rb'))

vocab_size = dm_prop.vocab_size
sos_idx = dm_prop.sos_indx
eos_idx = dm_prop.eos_indx
pad_idx = dm_prop.pad_indx
max_sequence_length = dm_prop.max_length
char2ind = dm_prop.char2ind

NLL = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_idx)

model = Model(vocab_size=vocab_size, embedding_size=30, rnn_type='gru', hidden_size=1024, word_dropout=0.0, latent_size=196,
                sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx, max_sequence_length=240, num_layers=1, prop_hidden_size=[256, 192, 128, 64, 32], 
                predict_prop = True, bidirectional=False, gpu_exist = True, NLL=NLL)

print('Loading model ...')
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model
print('Saving model ...')
torch.save(model.state_dict(), model_path)