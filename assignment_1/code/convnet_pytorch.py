"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import squeeze
import torch.nn as nn
from collections import OrderedDict

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """
    super().__init__()

    self.model = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(n_channels, 64, (3, 3), 1, 1)),
      ('batchnorm1', nn.BatchNorm2d(64)),
      ('relu1', nn.ReLU()),
      ('maxpool1', nn.MaxPool2d((3, 3), 2, 1)),

      ('conv2', nn.Conv2d(64, 128, (3, 3), 1, 1)),
      ('batchnorm2', nn.BatchNorm2d(128)),
      ('relu2', nn.ReLU()),
      ('maxpool2', nn.MaxPool2d((3, 3), 2, 1)),

      ('conv3_a', nn.Conv2d(128, 256, (3, 3), 1, 1)),
      ('batchnorm3_a', nn.BatchNorm2d(256)),
      ('relu3_a', nn.ReLU()),
      ('conv3_b', nn.Conv2d(256, 256, (3, 3), 1, 1)),
      ('batchnorm3_b', nn.BatchNorm2d(256)),
      ('relu3_b', nn.ReLU()),
      ('maxpool3', nn.MaxPool2d((3, 3), 2, 1)),

      ('conv4_a', nn.Conv2d(256, 512, (3, 3), 1, 1)),
      ('batchnorm4_a', nn.BatchNorm2d(512)),
      ('relu4_a', nn.ReLU()),
      ('conv4_b', nn.Conv2d(512, 512, (3, 3), 1, 1)),
      ('batchnorm4_b', nn.BatchNorm2d(512)),
      ('relu4_b', nn.ReLU()),
      ('maxpool4', nn.MaxPool2d((3, 3), 2, 1)),

      ('conv5_a', nn.Conv2d(512, 512, (3, 3), 1, 1)),
      ('batchnorm5_a', nn.BatchNorm2d(512)),
      ('relu5_a', nn.ReLU()),
      ('conv5_b', nn.Conv2d(512, 512, (3, 3), 1, 1)),
      ('batchnorm5_b', nn.BatchNorm2d(512)),
      ('relu5_b', nn.ReLU()),
      ('maxpool5', nn.MaxPool2d((3, 3), 2, 1)),

      ('avgpool', nn.AvgPool2d((1, 1), 1, 0)),
    ]))
    self.last_layer = nn.Linear(512, n_classes)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    out = self.model(x)
    return self.last_layer(squeeze(out))
