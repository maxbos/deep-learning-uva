################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
import pandas as pd

import sys
sys.path.append("..")

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    model_name = VanillaRNN if config.model_type == 'RNN' else LSTM
    model = model_name(
        config.input_length, config.input_dim, config.num_hidden,
        config.num_classes, config.batch_size, device
    )
    model.to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    accuracies = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        batch_targets = batch_targets.to(device)

        # Only for time measurement of step through network
        t1 = time.time()

        # zero the parameter gradients
        optimizer.zero_grad()

        ############################################################################
        # QUESTION: what happens here and why?
        # ANSWER: gradient clipping is applied to counteract the exploding gradient
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################
    
        # forward + backward + optimize
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        accuracy = (outputs.argmax(dim=1) == batch_targets).float().mean()
        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy.item(), loss
            ))

            accuracies.append(accuracy.item())

            # Stop the training on convergence
            if loss < 0.001:
                break

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    # Calculate the average achieved accuracy over the last 10 batches
    return np.mean(np.array(accuracies[-10:]))

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--num_runs', type=int, default=3, help="Number of training runs")

    config = parser.parse_args()

    # Train the model
    accuracies = []
    for r in range(config.num_runs):
        accuracies.append(train(config))

    out_dir = './out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    acc_df = pd.DataFrame(data={
        'model type': [config.model_type],
        'palindrome length': [config.input_length],
        # Average the accuracy over all performed runs
        'accuracy': [np.mean(np.array(accuracies))]
    })
    acc_df.to_csv('./out/accuracy_model-{}_pl-{}.csv'.format(config.model_type, config.input_length))
