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
        print('test')
        self.h_init = np.zeros(seq_length)
        self.W_hx = nn.Parameter()
        self.W_hh = nn.Parameter()
        self.W_ph = nn.Parameter()
        self.b_h = nn.Parameter()
        self.b_p = nn.Parameter()

    def forward(self, x):
        print('asfsafd')
        h_prev = self.h_init
        for t in range(self.seq_length):
            x_t = x[:, t]
            h_t = np.matmul(self.W_hx, x_t) + np.matmul(self.W_hh, h_prev) + self.b_h
        p_t = np.matmul(self.W_ph, h_t) + self.b_p
