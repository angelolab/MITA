import matplotlib.pyplot as plt
import os
from pathlib import Path
from itertools import groupby
from PIL import Image
import csv
import numpy as np
from skimage.transform import resize

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32

path = "models/"
print ("Initializing MIBINet")
net = MIBINet()

print ("Network setup done")
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_dataset = MIBIDataset('train_sim_pairs.npz', 'train_dissim_pairs.npz', None)
val_dataset = MIBIDataset('eval_sim_pairs.npz', 'eval_dissim_pairs.npz', None)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32,shuffle=True)

print ("Starting training")
for epoch in range(10):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [input_1, input_2, label]
        input_1, input_2, labels = data
        labels = torch.unsqueeze(labels, 1)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = net(input_1.float(), input_2.float())
        loss = criterion(output.float(), labels.float())
        print (loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            torch.save(net.state_dict(), path)
            running_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for i, data in enumerate(valloader, 0):
            input_1, input_2, labels = data
            labels = torch.unsqueeze(labels, 1)

            # Forward pass
            output = net(input_1.float(), input_2.float())
            loss = criterion(output.float(), labels.float())

            # Compute evaluation metrics
            total_loss += loss.item() * input_1.size(0)
            total_accuracy += accuracy(output, labels) * input_1.size(0)
            total_samples += input_1.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples


print('Finished Training')