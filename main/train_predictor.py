import pickle
import os
import sys
sys.path.append("..")
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import argparse
import warnings

from src.model_cov import Model
from src.model_predictor import Predictor_Model
from src.util import char_weight

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
# torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=30)
parser.add_argument('--rnn_type', type=str, default='gru')
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--word_dropout', type=float, default=0.0)
parser.add_argument('--latent_size', type=int, default=192)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--prop_hidden_size', type=list, default=[256, 192, 128, 64, 32])
parser.add_argument('--model_path', type=str, default='../checkpoints/vae_prop/checkpoint.pt')

parser.add_argument('--num_devices', type=int, default=4)
parser.add_argument('--epochs', type=int, default=30)

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
                predict_prop = False, bidirectional=False, gpu_exist = True, NLL=NLL)

print('Loading model ...')
model.load_state_dict(torch.load(args.model_path))
for p in model.parameters():
    p.requires_grad = False
print('Model: ', model)
model.eval()
model.cuda()

predictor_model = Predictor_Model(vae_model=model, prop_hidden_size=args.prop_hidden_size)
trainer = pl.Trainer(devices=args.num_devices, accelerator="gpu", strategy='ddp', max_epochs=args.epochs, logger=pl.loggers.CSVLogger('../logs', name='predictor_prop_log'),
                    callbacks=[LearningRateMonitor(logging_interval="epoch"),ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=10, dirpath='../checkpoints/predictor_prop', save_last=True)])

print('Training..')
trainer.fit(predictor_model, dm_prop)

# print('Saving..')
# model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).cuda()
# torch.save(model.state_dict(), '../checkpoints/vae_prop/checkpoint.pt')
# print('Model saved')