import matplotlib.pyplot as plt
import os
from pathlib import Path
from itertools import groupby
from PIL import Image
import csv
import numpy as np
from skimage.transform import resize
from utils.data_processing_utils import *
from utils.model import *
import torch.optim as optim
import torch
import torchvision.models as models
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import glob
from MIBIDataset import MIBIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MIBINet()
model = model.to(device)

# Load the saved state_dict
saved_state_dict = torch.load('~/models/ckpt_47.pt')

model.load_state_dict(saved_state_dict)
model.eval()

test_dataset = MIBIDataset('~/data/test_sim_pairs.npz', '~/data/test_dissim_pairs.npz', None)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

def accuracy(predictions, labels, threshold=0.5):
    binary_predictions = (predictions >= 0.5).float()
    correct = (binary_predictions == labels).sum().item()
    total = labels.size(0)

    # Compute accuracy
    accuracy = correct / total

    return accuracy

with torch.no_grad():
    total_accuracy = 0.0
    total_samples = 0

    for i, data in enumerate(test_loader, 0):
        input_1, input_2, labels = data
        labels = torch.unsqueeze(labels, 1)
        labels = labels.to(device)
        input_1 = input_1.to(device)
        input_2 = input_2.to(device)
        output = model(input_1.float(), input_2.float())
        output = output.to(device)
        total_accuracy += accuracy(output, labels) * input_1.size(0)
        total_samples += input_1.size(0)

    avg_accuracy = total_accuracy / total_samples

print("Test Accuracy:", avg_accuracy)