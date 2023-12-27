## Introduction

This is a small tool that uses the PyTorch framework, providing assistance in completing classification task using CNN.

Features: train model, print training process, save training files, plot figures, etc.

## Install

`pip install fau-tools`

## Usage

### import

The following code is recommended.

```python
import fau_tools
```

### quick start

The tutor will use a simple example to help you get started quickly!

**The following example uses Fau-tools to train a model in MNIST hand-written digits dataset.**

```python
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision

import fau_tools


# A simple CNN network
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1, 16, 3, 1, 1),  # -> (16, 28, 28)
      nn.ReLU(),
      nn.MaxPool2d(2),  # -> (16, 14, 14)

      nn.Conv2d(16, 32, 3, 1, 1),  # -> (32, 14, 14)
      nn.ReLU(),
      nn.MaxPool2d(2)  # -> (32, 7, 7)
    )
    self.output = nn.Linear(32 * 7 * 7, 10)


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
```

Now, we can run the python script, and the training process will be visualized as the following picture.

![training_visualization](github_attachment/training_visualization.png)

> Three files named `best.pth`, `scalars.csv` and `exp_info.txt` will be saved.
>
> The first file is the weight of trained model.
>
> The second file records scalar value changes in the training process.
>
> The third file saves information about the experiment.

---

The above is the primary usage of this tool, but there are also some other snazzy features, which will be introduced later. [TODO]

## END

Hope you could like it! And welcome issues and pull requests.
