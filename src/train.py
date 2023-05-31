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

print ("Initializing MIBINet")
net = MIBINet()

print ("Network setup done")
criterion = ContrastiveLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
sim_csv_file = "sim_samples.csv"
dissim_csv_file = "dissim_samples.csv"
dataset = MIBIDataset(sim_csv_file, dissim_csv_file, None)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=32,shuffle=True)

print ("Starting training")
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [input_1, input_2, label]
        input_1, input_2, labels = data
        labels = torch.unsqueeze(labels, 1)
        # zero the parameter gradients
        optimizer.zero_grad()
        print (input_1.shape)
        # forward + backward + optimize
        outputs = net(input_1.float(), input_2.float())
        loss = criterion(outputs, labels.float())
        #print (loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            torch.save(net.state_dict(), path)
            running_loss = 0.0

print('Finished Training')