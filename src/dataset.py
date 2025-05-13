from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from src.util import *
from src.data_prepration import dataset_building

class MolDataModule(pl.LightningDataModule):
    def __init__(self, datafile, properties, max_length, batch_size):
        super().__init__()
        self.datafile = datafile
        self.properties = properties
        self.max_length = max_length
        self.batch_size = batch_size
        print('############### data loading ######################')
        data = text2dict_zinc(self.datafile, self.properties)
        print("Number of SMILES >>>> ", len(data['SMILES']))
        data_ = smi_postprocessing(data, self.max_length)
        print("Set Max_Length >>>>", self.max_length)
        print("Number of SELFIES >>>> ", len(data_['SELFIES']))
        data_type = 'smiles_properties'
        self.char2ind, self.ind2char, self.sos_indx, self.eos_indx, self.pad_indx = dictionary_build(data_['SELFIES'])
        self.dataset = dataset_building(self.char2ind, data_, self.max_length, data_type)
        self.train_data, self.valid_data = random_split(self.dataset, [int(round(len(self.dataset) * 10/11)), int(round(len(self.dataset) * 1/11))])
        self.vocab_size = len(self.char2ind)
        

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, drop_last=True, num_workers=8, pin_memory=True)
