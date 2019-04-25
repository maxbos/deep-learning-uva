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

        self.h_init = torch.zeros(batch_size, num_hidden)
        self.W_hx = self._parameter(input_dim, num_hidden)
        self.W_hh = self._parameter(num_hidden, num_hidden)
        self.W_ph = self._parameter(num_hidden, num_classes)
        self.b_h = self._parameter(num_hidden)
        self.b_p = self._parameter(num_classes)

    def _parameter(self, *params):
        y = params[1] if len(params) > 1 else 1
        stdv = 1. / np.sqrt(y)
        return nn.Parameter(torch.empty(*params).uniform_(-stdv, stdv))

    def forward(self, x):
        h_prev = self.h_init
        for t in range(self.seq_length):
            # Retrieve a sequence of `input_dim` starting at timestamp `t` for all samples
            # resulting in a matrix of size (batch_size x input_dim)
            x_t = x[:, t:(t+self.input_dim)]
            h_t = torch.tanh(x_t @ self.W_hx + h_prev @ self.W_hh + self.b_h)
            p_t = h_t @ self.W_ph + self.b_p
            h_prev = h_t
        return p_t
