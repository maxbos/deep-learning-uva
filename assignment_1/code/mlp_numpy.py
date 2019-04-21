"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """
    # The network is structured as a sequence of modules
    self.modules = []
    # Add the input layer and each hidden layer with their activation
    # modules to the network
    in_features = n_inputs
    for i in range(len(n_hidden)):
      self.modules.extend([
        LinearModule(in_features, n_hidden[i]),
        ReLUModule(),
      ])
      in_features = n_hidden[i]
    # Add the last output layer which has as input the neurons of the
    # last hidden layer and as output the number of classes, over which
    # softmax is calculated.
    self.modules.extend([
      LinearModule(in_features, n_classes),
      SoftMaxModule(),
    ])

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

    out = x
    for i in range(len(self.modules)):
      out = self.modules[i].forward(out)

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    for i in reversed(range(len(self.modules))):
      dout = self.modules[i].backward(dout)

    return
