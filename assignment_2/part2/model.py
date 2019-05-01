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

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0',
                 embedding_dim=30):

        super(TextGenerationModel, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim).to(device)
        self.lstm = nn.LSTM(
            embedding_dim, lstm_num_hidden, lstm_num_layers,
        ).to(device)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size).to(device)

    def forward(self, x, states=None):
        embeds = self.embeddings(x.to(self.device))
        output, (hn, cn) = self.lstm(embeds, states)
        output = self.linear(output)
        return output, (hn, cn)

    def predict(self, x, states, temperature=1.0):
        """Predict the next character."""
        output, (hn, cn) = self.forward(x, states)
        distribution = nn.functional.softmax(output.squeeze()/temperature, dim=0)
        predicted = torch.multinomial(distribution, 1)
        return predicted, (hn, cn)
