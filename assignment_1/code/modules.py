"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    self.params = {
      'weight': np.random.normal(loc=0, scale=0.0001, size=(out_features, in_features)),
      'bias': np.zeros((out_features, 1)),
    }
    self.grads = {
      'weight': np.zeros((out_features, in_features)),
      'bias': np.zeros((out_features, 1)),
    }

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    # Store the input for usage in the backward step
    self.x = x
    # We transpose the input `x` since it is a matrix of size `n_samples x n_features`
    # and we want to calculate the weighted inputs.
    out = np.matmul(self.params['weight'], x.T) + self.params['bias']
    
    # We transpose the result back to the size `n_samples x n_features` for future forward
    # calls.
    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    # Calculate the gradients of the current module
    dx = np.matmul(dout, self.params['weight'])
    # Calculate the gradients w.r.t. to the weights of the current layer
    self.grads['weight'] = np.matmul(dout.T, self.x)
    # Reshape the `dout` matrix, since this is a matrix of gradients per sample,
    # we want to perform a matrix multiplication between one sample and one `ones` vector.
    self.grads['bias'] = np.matmul(dout.T, np.ones((dout.shape[0], 1)))
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    self.x_mask = x > 0
    out = x.clip(min=0)

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    dx = np.multiply(dout, self.x_mask)  

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    # Get the maximum value at each row (each sample)
    b = x.max(axis=1)[:, None]
    # Use the Exp-normalize trick, so from each value subtract the maximum value
    # in its row (same sample) and calculate the exponential
    y = np.exp(x - b)
    # Sum the exponential values over each row (which is one sample),
    # and divide each exponential value by the summation of its row
    out = y / y.sum(axis=1)[:, None]
    self.n_outputs = x.shape[1]
    self.out = out

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    # Create a 3d-tensor where the first dimension is the number of samples,
    # and the value of sample is repeated on the x axis, to create a 2d matrix 
    # for each sample.
    # softmax_grads_3d = np.repeat(self.out[:, :, np.newaxis], self.n_outputs, axis=2)
    grads_map = np.dstack([self.out] * self.n_outputs)
    eye_matrix = np.identity(self.n_outputs)
    softmax_grads = np.einsum('ik, ijk -> ijk', self.out, np.subtract(eye_matrix, grads_map))
    dx = np.einsum('ij, ijk -> ik', dout, softmax_grads)

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    # For every neuron in the input that corresponds to the neuron for
    # the actual correct target, we calculate the negative log.
    # Do this by first performing element-wise multiplication of x and y,
    # this yields a matrix with only values at the positions where y = 1.
    # Finally, calculate the average loss from all individual losses.
    out = (-np.log(np.multiply(x, y).sum(axis=1))).mean()

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    dx = -np.divide(y, x)/len(x)

    return dx
