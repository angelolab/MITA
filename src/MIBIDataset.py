from utils import *
import torch
from torch.utils.data import Dataset

class MIBIDataset(Dataset):
    def __init__(self, similar_samples=None, dissimilar_samples=None, transform=None):
        self.sim_df=pd.read_csv(similar_samples)
        self.sim_df.columns =["Tile_1","Tile_2","Label"]
        self.dissim_df=pd.read_csv(dissimilar_samples)
        self.dissim_df.columns =["Tile_1","Tile_2","Label"]
        self.transform = transform

    def __getitem__(self, index):
        if index < len(self.similar_samples):
            # Fetch a pair of similar samples
            sample1, sample2, label = self.sim_df[index]
        else:
            # Fetch a pair of dissimilar samples
            index -= len(self.similar_samples)
            sample1, sample2, label = self.dissim_df[index]

        return sample1, sample2, label

    def __len__(self):
        return len(self.similar_samples) + len(self.dissimilar_samples)