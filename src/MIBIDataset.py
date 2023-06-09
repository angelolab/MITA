from utils import *
import torch
from torch.utils.data import Dataset
import numpy as np
class MIBIDataset(Dataset):
    def __init__(self, sim_df=None, dissim_df=None, transform=None):
        self.sim_df = np.load(sim_df)
        self.dissim_df = np.load(dissim_df)
        self.sim_df_files = self.sim_df.files
        self.dissim_df_files = self.dissim_df.files

    def __getitem__(self, index):
        if index < len(self.sim_df_files):
            # Fetch a pair of similar samples
            elem = self.sim_df_files[index]
            sample1, sample2 = self.sim_df[elem]
            label = 0
            
        else:
            # Fetch a pair of dissimilar samples
            index -= len(self.sim_df)
            elem = self.dissim_df_files[index]
            sample1, sample2 = self.dissim_df[elem]
            label = 1
        
        sample1 = torch.tensor(sample1)
        sample1 = sample1.permute(2, 1, 0)
        sample2 = torch.tensor(sample2)
        sample2 = sample2.permute(2, 1, 0)
        
        return sample1, sample2, torch.tensor(label)

    def __len__(self):
        return len(self.sim_df_files) + len(self.dissim_df_files)