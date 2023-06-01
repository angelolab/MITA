from utils import *
import torch
from torch.utils.data import Dataset

class MIBIDataset(Dataset):
    def __init__(self, sim_df=None, dissim_df=None, transform=None):
        self.sim_df = sim_df
        self.dissim_df = dissim_df
        self.transform = transform

    def __getitem__(self, index):
        if index < len(self.sim_df):
            # Fetch a pair of similar samples
            sample1, sample2, label = self.sim_df.iloc[index]
            
        else:
            # Fetch a pair of dissimilar samples
            index -= len(self.sim_df)
            sample1, sample2, label = self.dissim_df.iloc[index]
        
        sample1 = torch.tensor(sample1)
        sample1 = sample1.permute(2, 1, 0)
        sample2 = torch.tensor(sample2)
        sample2 = sample2.permute(2, 1, 0)
        
        return sample1, sample2, torch.tensor(label)

    def __len__(self):
        return len(self.sim_df) + len(self.dissim_df)