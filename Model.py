import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import os
import sys
from torch.nn.utils import clip_grad_norm_
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem


class Generator(nn.Module):
    def __init__(self, voc, emb_size, hidden_size, num_layers, dropout):
        super(Generator, self).__init__()
        self.voc = voc
        self.embedding_layer = nn.Embedding(voc.vocab_size, emb_size, padding_idx=voc.vocab['PAD'])
        self.lstm_layer = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size, voc.vocab_size)

    def forward(self, x, lengths, states=None):
        if not hasattr(self, '_flattened'):
            self.lstm_layer.flatten_parameters()
            setattr(self, '_flattened', True)
        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True)
        x, states = self.lstm_layer(x, states)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)
        return x, lengths, states


class Discriminator(nn.Module):
    def __init__(self, voc, emb_size, convs, dropout=0):
        super(Discriminator, self).__init__()
        self.embedding_layer = nn.Embedding(voc.vocab_size, emb_size, padding_idx=voc.vocab['PAD'])
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, f, kernel_size=(n, emb_size)) for f, n in convs])
        sum_filters = sum([f for f, _ in convs])
        self.highway_layer = nn.Linear(sum_filters, sum_filters)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(sum_filters, 1)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.unsqueeze(1)
        convs = [F.elu(conv_layer(x)).squeeze(3) for conv_layer in self.conv_layers]
        x = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in convs]
        x = torch.cat(x, dim=1)
        h = self.highway_layer(x)
        t = torch.sigmoid(h)
        x = t * F.elu(h) + (1 - t) * x
        x = self.dropout_layer(x)
        out = self.output_layer(x)
        return out




