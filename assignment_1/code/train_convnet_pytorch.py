"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import time
import torch
import csv

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 500
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """
  predictions = predictions.data.cpu().numpy()
  accuracy = (np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)).mean()
  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Get cifar10 data
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  ## Testing data
  x_test, y_test = cifar10['test'].images, cifar10['test'].labels
  # Prepare the dimensions of the network input and output
  _, x_channels, x_height, x_width = x_test.shape
  # The number of different classes in our y target vector
  _, n_classes = y_test.shape
  ## Create a ConvNet instance
  model = ConvNet(x_channels, n_classes)
  model.to(DEVICE)
  cross_entropy = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

  evaluation_data = []

  for step in range(FLAGS.max_steps):
    # Get the next training batch.
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    # Call zero_grad before `loss.backward` to not accumulate the gradients from
    # multiple passes
    optimizer.zero_grad()
    # Perform a forward pass through our network.
    out = model.forward(torch.from_numpy(x).float().to(DEVICE))
    # Calculate the cross entropy loss for our prediction.
    loss = cross_entropy(out, torch.from_numpy(y).float().to(DEVICE).argmax(dim=1))
    # Perform backward propagation
    loss.backward()
    # Update the weights using the Adam optimizer
    optimizer.step()
    out.detach()
    # Only evaluate the model on the whole test set each `eval_freq` iterations
    if (step % FLAGS.eval_freq == 0):
      # Calculate train accuracy
      train_accuracy = accuracy(out, y)
      print('Train accuracy:', train_accuracy)
      # Perform a forward propagation using the test set
      test_out = model.forward(torch.from_numpy(x_test).float().to(DEVICE))
      test_loss = cross_entropy.forward(test_out, torch.from_numpy(y_test).float().to(DEVICE).argmax(dim=1))
      test_accuracy = accuracy(test_out, y_test)
      print('Test accuracy:', test_accuracy)
      evaluation_data.extend([
        ['train loss', step, loss.item()],
        ['train accuracy', step, train_accuracy],
        ['test loss', step, test_loss.item()],
        ['test accuracy', step, test_accuracy],
      ])
      test_out.detach()

  eval_dir = './eval/cnn_pytorch/'
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  ## Write evaluation data to csv
  fname = str(time.time())
  with open(eval_dir+fname+'.csv', 'w') as outcsv:
      writer = csv.writer(outcsv)
      writer.writerow(['label', 'step', 'value'])
      writer.writerows(evaluation_data)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()