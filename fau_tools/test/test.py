from typing import override

import torch
import torch.utils.data as tdata
import torchvision
from torch import nn

import fau_tools


# A simple CNN network
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv: nn.Sequential = nn.Sequential(
      nn.Conv2d(1, 16, 3, 1, 1),  # -> (16, 28, 28)
      nn.ReLU(),
      nn.MaxPool2d(2),  # -> (16, 14, 14)

      nn.Conv2d(16, 32, 3, 1, 1),  # -> (32, 14, 14)
      nn.ReLU(),
      nn.MaxPool2d(2)  # -> (32, 7, 7)
    )
    self.output: nn.Linear = nn.Linear(32 * 7 * 7, 10)


  @override
  def forward(self, x):
    x = self.conv(x)
    x = x.flatten(1)
    return self.output(x)


# Hyper Parameters definition
total_epoch = 10
lr = 1E-2
batch_size = 1024

# Load dataset
train_data      = torchvision.datasets.MNIST('datasets', True, torchvision.transforms.ToTensor(), download=True)
test_data       = torchvision.datasets.MNIST('datasets', False, torchvision.transforms.ToTensor())
train_data.data = train_data.data[:6000]  # mini data
test_data.data  = test_data.data[:2000]  # mini data

# Get data loader
train_loader = tdata.DataLoader(train_data, batch_size, True)
test_loader  = tdata.DataLoader(test_data, batch_size)

# Initialize model, optimizer and loss function
model = CNN()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

# Train!
fau_tools.TaskRunner(model, train_loader, test_loader, loss_function, optimizer, total_epoch, exp_path="MNIST").train()
