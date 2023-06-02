import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class MIBINet(nn.Module):

    def __init__(self):
        super(MIBINet, self).__init__()
        # Shared convolutional layers
        self.resnet = models.resnet18(pretrained=False)
        # Remove the last fully connected layer of ResNet-18
        self.resnet.fc = nn.Identity()
        # Add a new fully connected layer for classification
        self.fc = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward_once(self, x):
        output = self.resnet(x)
        return output

    def forward(self, input1, input2):
        # Pass the inputs through the ResNet-18 backbone
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        
        # Concatenate the embeddings
        combined = torch.cat((out1, out2), dim=1)
        
        # Pass the combined embeddings through the classification layer
        output = self.fc(combined)
        output = self.sigmoid(output)

        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive