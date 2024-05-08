# This source code contains the PyTorch module for our modified ResNet18.
#
# Author: Matt Krol

import torch
import torch.nn as nn
import torch.nn.functional as F

from cpconv2d import CPConv2d


torch.manual_seed(0)
torch.cuda.manual_seed(0)


def mkconv(in_channels, out_channels, kernel_size=3, stride=1, padding=0,
           bias=False, rank=0, last_rank=0, mode=4):
    if rank > 0:
        return CPConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            rank=rank,
            last_rank=last_rank,
            mode=mode
        )
    else:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )


class IdentityMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityMap, self).__init__()
        if in_channels == out_channels:
            self.lambd = lambda x: x
        elif in_channels < out_channels:
            padding = (0, 0, 0, 0, out_channels//4, out_channels//4)
            self.lambd = lambda x: F.pad(x[:, :, ::2, ::2], padding, 'constant', 0)
        else:
            raise ValueError('in_channels cannot be >= to out_channels')


    def forward(self, x):
        return self.lambd(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 rank=0, last_rank=0, mode=4):
        super(ResBlock, self).__init__()

        self.conv1 = mkconv(in_channels, out_channels, kernel_size=3,
                            stride=stride, padding=1, bias=False,
                            rank=rank, last_rank=last_rank, mode=mode)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = mkconv(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, bias=False,
                            rank=rank, last_rank=last_rank, mode=mode)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = IdentityMap(in_channels, out_channels)


    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    """
    Standard ResNet18 with some modifications for smaller images.
    """
    def __init__(self, outputs, rank=0, last_rank=0, mode=4):
        super(ResNet18, self).__init__()
        self.conv1 = mkconv(3, 64, kernel_size=7, stride=1,
                            padding=3, bias=False, rank=rank,
                            last_rank=last_rank, mode=mode)
        self.bn1 = nn.BatchNorm2d(64)

        self.res1 = ResBlock(64, 64, stride=1, rank=rank,
                             last_rank=last_rank, mode=mode)
        self.res2 = ResBlock(64, 64, stride=1, rank=rank,
                             last_rank=last_rank, mode=mode)

        self.res3 = ResBlock(64, 128, stride=2, rank=rank,
                             last_rank=last_rank, mode=mode)
        self.res4 = ResBlock(128, 128, stride=1, rank=rank,
                             last_rank=last_rank, mode=mode)

        self.res5 = ResBlock(128, 256, stride=2, rank=rank,
                             last_rank=last_rank, mode=mode)
        self.res6 = ResBlock(256, 256, stride=1, rank=rank,
                             last_rank=last_rank, mode=mode)

        self.res7 = ResBlock(256, 512, stride=2, rank=rank,
                             last_rank=last_rank, mode=mode)
        self.res8 = ResBlock(512, 512, stride=1, rank=rank,
                             last_rank=last_rank, mode=mode)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, outputs)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.res3(x)
        x = self.res4(x)

        x = self.res5(x)
        x = self.res6(x)

        x = self.res7(x)
        x = self.res8(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
