# This file contains the PyTorch modules for the CP convolutional layers.
#
# Author: Matt Krol

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
tl.set_backend('pytorch')


torch.manual_seed(0)
torch.cuda.manual_seed(0)


class CPFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 rank, mode=4, last_rank=0):
        super(CPFilter, self).__init__()

        self.rank = rank
        self.last_rank = last_rank
        self.mode = mode

        # Store CP vectors in matrices.
        self.factors = nn.ParameterList()
        if mode == 3:
            self.factors.append(nn.Parameter(torch.randn((kernel_size, rank))))
            self.factors.append(nn.Parameter(torch.randn((kernel_size, rank))))
            self.factors.append(nn.Parameter(torch.randn((in_channels, rank))))
        elif mode == 4:
            self.factors.append(nn.Parameter(torch.randn((kernel_size, rank))))
            self.factors.append(nn.Parameter(torch.randn((kernel_size, rank))))
            self.factors.append(nn.Parameter(torch.randn((in_channels, rank))))
            self.factors.append(nn.Parameter(torch.randn((out_channels, rank))))
        else:
            raise ValueError('invalid mode value for CPFilter')

        # Create selector weights.
        if last_rank > 0:
            self.s = nn.Parameter(torch.ones((last_rank,)))


    def forward(self):
        if self.last_rank > 0:
            spadded = F.pad(self.s, (0, self.rank - self.last_rank),
                            mode='constant', value=1.0)
        else:
            spadded = None
        weight = tl.cp_to_tensor((spadded, self.factors))
        weight = torch.permute(weight, (2, 1, 0) if self.mode == 3 else (3, 2, 1, 0))
        return weight


class CPConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=0,
                 last_rank=0, stride=1, padding=0, bias=True, mode=4):
        super(CPConv2d, self).__init__()
        
        self.rank = rank
        self.last_rank = last_rank
        self.stride = stride
        self.padding = padding
        self.mode = mode

        if mode == 3:
            self.filters = nn.ModuleList([ CPFilter(in_channels, out_channels, kernel_size, rank, mode=3, last_rank=last_rank) for i in range(out_channels) ])
        elif mode == 4:
            self.filters = nn.ModuleList([ CPFilter(in_channels, out_channels, kernel_size, rank, mode=4, last_rank=last_rank) ])
        else:
            raise ValueError('invalid value for mode')

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None


    def forward(self, x):
        if self.mode == 3:
            # This returns S filters of shape C x l x l.
            filter_weights = [ f() for f in self.filters ]
            # Stack filters to get a tensor of shape S x C x l x l for PyTorch.
            weight = torch.stack(filter_weights, dim=0)
        else:
            # This is for the 4way approach.
            weight = self.filters[0]()
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride,
                     padding=self.padding)
        return x
