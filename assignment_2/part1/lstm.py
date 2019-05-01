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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.device = device

        self.c_init = torch.zeros(batch_size, num_hidden, device=device)
        self.h_init = torch.zeros(batch_size, num_hidden, device=device)
        self.W_gx = self._parameter(input_dim, num_hidden)
        self.W_gh = self._parameter(num_hidden, num_hidden)
        self.W_ix = self._parameter(input_dim, num_hidden)
        self.W_ih = self._parameter(num_hidden, num_hidden)
        self.W_fx = self._parameter(input_dim, num_hidden)
        self.W_fh = self._parameter(num_hidden, num_hidden)
        self.W_ox = self._parameter(input_dim, num_hidden)
        self.W_oh = self._parameter(num_hidden, num_hidden)
        self.W_ph = self._parameter(num_hidden, num_classes)
        self.b_g = self._parameter(num_hidden)
        self.b_i = self._parameter(num_hidden)
        self.b_f = self._parameter(num_hidden)
        self.b_o = self._parameter(num_hidden)
        self.b_p = self._parameter(num_classes)

    def _parameter(self, *params):
        y = params[1] if len(params) > 1 else 1
        stdv = 1. / np.sqrt(y)
        return nn.Parameter(torch.empty(*params, device=self.device).uniform_(-stdv, stdv))

    def forward(self, x):
        x = x.to(self.device)
        c_prev = self.c_init
        h_prev = self.h_init
        for t in range(self.seq_length):
            x_t = x[:, t:(t+self.input_dim)]
            g_t = torch.tanh(x_t @ self.W_gx + h_prev @ self.W_gh + self.b_g)
            i_t = torch.sigmoid(x_t @ self.W_ix + h_prev @ self.W_ih + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_fx + h_prev @ self.W_fh + self.b_f)
            o_t = torch.sigmoid(x_t @ self.W_ox + h_prev @ self.W_oh + self.b_o)
            c_t = g_t * i_t + c_prev * f_t
            h_t = torch.tanh(c_t) * o_t
            p_t = h_t @ self.W_ph + self.b_p
            c_prev = c_t
            h_prev = h_t
        return p_t
