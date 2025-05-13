import os
import sys
import pickle
sys.path.append("..")
import argparse

from src.dataset import MolDataModule

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='../data/raw/dft_predictions.csv')
parser.add_argument('--properties', type=list, default=["HLGap", "IP", "EA", "OpticalGap"])
parser.add_argument('--max_length', type=int, default=240)
parser.add_argument('--batch_size', type=int, default=512)

if os.path.exists('../data/processed/dm_prop.pkl'):
    os.remove('../data/processed/dm_prop.pkl')

args = parser.parse_args()
print('Preprocessing..')
dm = MolDataModule(args.data_file, args.properties, args.max_length, args.batch_size)
pickle.dump(dm, open('../data/processed/dm_prop.pkl', 'wb'))
print('Done!')