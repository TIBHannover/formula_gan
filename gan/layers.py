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


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class SNSelfAttention2d(nn.Module):
    def __init__(self, in_channels):
        super(SNSelfAttention2d, self).__init__()
        # Channel multiplier
        self.in_channels = in_channels
        self.theta = nn.utils.spectral_norm(
            nn.Conv2d(self.in_channels, self.in_channels // 8, kernel_size=1, padding=0, bias=False)
        )
        self.phi = nn.utils.spectral_norm(
            nn.Conv2d(self.in_channels, self.in_channels // 8, kernel_size=1, padding=0, bias=False)
        )
        self.g = nn.utils.spectral_norm(
            nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=1, padding=0, bias=False)
        )
        self.o = nn.utils.spectral_norm(
            nn.Conv2d(self.in_channels // 2, self.in_channels, kernel_size=1, padding=0, bias=False)
        )

        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.in_channels // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.in_channels // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class BigGANGeneratorBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, emb_channels, upsample=None, bn_eps=1e-5, bn_momentum=0.1,
    ):
        super(BigGANGeneratorBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels

        # upsample layers
        if upsample is None:
            self.upsample = functools.partial(F.interpolate, scale_factor=2)
        else:
            self.upsample = upsample

        self.activation = nn.ReLU()

        # Conv layers
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=[3, 3], padding=[1, 1])
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=[3, 3], padding=[1, 1])
        )
        self.learnable_sc = in_channels != out_channels or self.upsample is not None
        if self.learnable_sc:
            self.conv_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        # Batchnorm layers
        self.bn1 = ProjectionBatchNorm2d(in_channels, emb_channels=emb_channels)
        self.bn2 = ProjectionBatchNorm2d(out_channels, emb_channels=emb_channels)

    def forward(self, x, emb):
        h = self.activation(self.bn1(x, emb))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, emb))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


# Residual block for the discriminator
class BigGANDiscriminatorBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, downsample_first=False, skip_first_act=False, activation=None, downsample=None,
    ):
        super(BigGANDiscriminatorBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        wide = True
        self.hidden_channels = self.out_channels if wide else self.in_channels

        self.skip_first_act = skip_first_act
        self.downsample_first = downsample_first

        if activation is None:
            self.activation = nn.ReLU(inplace=False)
        else:
            self.activation = activation

        if downsample is None:
            self.downsample = nn.AvgPool2d(2)
        else:
            self.downsample = downsample

        # Conv layers
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=[3, 3], padding=[1, 1])
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=[3, 3], padding=[1, 1])
        )
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def shortcut(self, x):
        if not self.downsample_first:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.skip_first_act:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = self.activation(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)
