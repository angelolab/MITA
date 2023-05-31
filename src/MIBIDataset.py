from utils import *
import torch
from torch.utils.data import Dataset

class MIBIDataset(Dataset):
    def __init__(self, similar_samples, dissimilar_samples):
        self.similar_samples = similar_samples
        self.dissimilar_samples = dissimilar_samples

    def __getitem__(self, index):
        if index < len(self.similar_samples):
            # Fetch a pair of similar samples
            sample1, sample2 = self.similar_samples[index]
            label = torch.tensor(0)  # Similar pair label is 0
        else:
            # Fetch a pair of dissimilar samples
            index -= len(self.similar_samples)
            sample1, sample2 = self.dissimilar_samples[index]
            label = torch.tensor(1)  # Dissimilar pair label is 1

        return sample1, sample2, label

    def __len__(self):
        return len(self.similar_samples) + len(self.dissimilar_samples)