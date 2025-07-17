import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

class QuantLeNet(nn.Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)  # fixed input features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(F.relu(self.conv1(x)))  # output: [B, 6, 12, 12]
        x = self.pool(F.relu(self.conv2(x)))  # output: [B, 16, 4, 4]
        x = x.view(-1, 16 * 4 * 4)  # = 256
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x
