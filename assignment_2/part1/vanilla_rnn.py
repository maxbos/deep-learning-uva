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

import torch
import torch.nn as nn
import numpy as np

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim

        self.h_init = torch.zeros(num_hidden, batch_size)
        self.W_hx = nn.Parameter(torch.empty(num_hidden, input_dim).uniform_(0, 1))
        self.W_hh = nn.Parameter(torch.empty(num_hidden, num_hidden).uniform_(0, 1))
        self.W_ph = nn.Parameter(torch.empty(num_classes, num_hidden).uniform_(0, 1))
        self.b_h = nn.Parameter(torch.empty(num_hidden, input_dim).uniform_(0, 1))
        self.b_p = nn.Parameter(torch.empty(num_classes, input_dim).uniform_(0, 1))

    def forward(self, x):
        h_prev = self.h_init
        for t in range(self.seq_length):
            # Retrieve a sequence of `input_dim` starting at timestamp `t` for all samples
            # resulting in a matrix of size (input_dim x batch_size)
            x_t = torch.transpose(x[:, t:(t+self.input_dim)], 0, 1)
            h_t = torch.tanh(self.W_hx @ x_t + self.W_hh @ h_prev + self.b_h)
            p_t = self.W_ph @ h_t + self.b_p
        return torch.transpose(p_t, 0, 1)
