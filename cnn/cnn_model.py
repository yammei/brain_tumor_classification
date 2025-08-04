import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 3 x 3 kernel size with 1 padding retrieving 16 layers of learned features using an rgb image's 3 values
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # retrieve max value in 2 x 2 window to reduce compute load
        self.pool = nn.MaxPool2d(2, 2)
        # produce another set of learned features from the previous convolution
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # linear transform/flatten 3d output of convolution into 128 neurons 
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        # linear transform into 2 neurons
        self.fc2 = nn.Linear(128, 4)

    # fully connected network
    def forward(self, x):
        # pass through convolution configuration and pool
        x = self.pool(F.relu(self.conv1(x)))
        # 2nd convolution pass and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # flatten convolution's 3d feature map (batch size, features)
        x = x.view(-1, 32 * 16 * 16)
        # relu activation for first linear transformation
        x = F.relu(self.fc1(x))
        # transform/reduce neuron output to 2 for classification
        x = self.fc2(x)
        return x