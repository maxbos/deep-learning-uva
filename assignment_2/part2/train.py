# MIT License
#
# Copyright (c) 2017 Tom Runia
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
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

import sys
sys.path.append("..")

from part2.dataset import TextDataset
from part2.model import TextGenerationModel


def sample_text(model, generation_length, pretext, dataset, device, temperatures=[0, .5, 1., 2.]):
    with torch.no_grad():
        # Convert the `pretext` to integers
        if pretext:
            xs = [torch.LongTensor([[dataset._char_to_ix[ch]]]) for ch in pretext]
        # If no start is specified, start with a random character
        else:
            xs = [torch.randint(high=dataset.vocab_size, size=(1, 1), device=device)]
        # Add the first characters to the text for each temperature
        text = {temperature: [x.item() for x in xs] for temperature in temperatures}
        # Add the last character Tensor to the temperature data
        temp_data = {temperature: [xs[-1], None] for temperature in temperatures}
        # Forward through the whole start string, and append the last generated char
        if pretext:
            for i, x in enumerate(xs):
                for temperature in temperatures:
                    # Use the actual character as input, and the previous hidden state
                    x, states = model.predict(x.view(1, -1), temp_data[temperature][1], temperature)
                    temp_data[temperature] = [x, states]
                    # Only append the final generated character to the text
                    if i == len(xs)-1:
                        text[temperature].append(x.item())
        # Generate the rest of the text
        for _ in range(generation_length-1):
            for temperature in temperatures:
                x, states = model.predict(
                    temp_data[temperature][0].view(1, -1), temp_data[temperature][1], temperature
                )
                temp_data[temperature] = [x, states]
                text[temperature].append(x.item())
    return [[temp, dataset.convert_to_string(text[temp])] for temp in text]


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size, config.seq_length, dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden, lstm_num_layers=config.lstm_num_layers,
        device=device, embedding_dim=config.embedding_dim,
    )
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

    eval_results = []
    generated_text = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()
        
        batch_inputs = torch.stack(batch_inputs).to(device)
        batch_targets = torch.stack(batch_targets).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        # forward + backward + optimize
        outputs, _ = model(batch_inputs)
        # Reshape the input to be of size `seq_length` x `n_classes`
        loss = criterion(outputs.transpose(2, 1), batch_targets)
        # Get the index of the charachter with the maximum value for each timestep
        # and sample, and compare this to the true targets
        accuracy = (outputs.argmax(dim=-1) == batch_targets).float().mean()
        loss.backward(retain_graph=True)
        scheduler.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:
            print("[{}] Train Step {:.0f}/{:.0f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy.item(), loss.item()
            ))
            # Add the accuracy and loss values for the current step to the evaluation history
            eval_results.extend([
                ['accuracy', step, accuracy.item()],
                ['loss', step, loss.item()],
            ])

        if step % config.sample_every == 0:
            # Append a sentence that is generated
            generated_text.extend(sample_text(model, config.gen_length, config.gen_pretext, dataset, device))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    # Sample the text one last time with the final model
    generated_text.extend(sample_text(model, config.gen_length, config.gen_pretext, dataset, device))

    out_dir = './out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Store the evaluation results
    eval_df = pd.DataFrame(eval_results, columns=['label', 'step', 'value'])
    eval_df.to_csv('./out/eval_results_{}.csv'.format(config.outfile_suffix))

    # Store the generated text
    text_df = pd.DataFrame(generated_text, columns=['temperature', 'text'])
    text_df.to_csv('./out/generated_text_{}.csv'.format(config.outfile_suffix))


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--embedding_dim', type=str, default=30, help="Number of dimensions in word embedding")

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--gen_length', type=int, default=30, help='Length of generated text')
    parser.add_argument('--gen_pretext', type=str, default=None, help='Starting string for the generated text')
    parser.add_argument('--outfile_suffix', type=str, default=None, help='String added to the end of an outfilename')

    config = parser.parse_args()

    # Train the model
    train(config)
