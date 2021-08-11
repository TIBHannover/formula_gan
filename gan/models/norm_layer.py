import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import functools


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super(ConditionalBatchNorm2d, self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels, affine=False, **kwargs)
        self.embed = nn.Embedding(num_classes, in_channels * 2)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data[:, :in_channels].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, in_channels:].zero_()  # Initialise bias at 0

    def forward(self, x, c):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.in_channels, 1, 1) * out + beta.view(-1, self.in_channels, 1, 1)
        return out


class ProjectionBatchNorm2d(nn.Module):
    def __init__(self, in_channels, emb_channels, **kwargs):
        super(ProjectionBatchNorm2d, self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels, affine=False, **kwargs)
        self.linear = nn.Linear(emb_channels, in_channels * 2)
        self.linear.weight.data[:, :in_channels] = 1
        self.linear.weight.data[:, in_channels:] = 0

    def forward(self, x, emb):
        out = self.bn(x)
        emb = self.linear(emb)
        gamma, beta = emb.chunk(2, 1)
        out = gamma.view(-1, self.in_channels, 1, 1) * out + beta.view(-1, self.in_channels, 1, 1)
        return out

