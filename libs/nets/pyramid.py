from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


class PyramidFPN(nn.Module):

    def __init__(self, in_channels, f_keys, num_channels=256):
        super(PyramidFPN, self).__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        # only support f_keys = ['C2', 'C3', 'C4', 'C5']
        self.with_c2 = 'C2' in f_keys

        if self.with_c2:
            self.lateral_c2 = nn.Conv2d(in_channels[0], num_channels, 1, 1, padding=0, bias=True)
            self.lateral_c3 = nn.Conv2d(in_channels[1], num_channels, 1, 1, padding=0, bias=True)
            self.lateral_c4 = nn.Conv2d(in_channels[2], num_channels, 1, 1, padding=0, bias=True)
            self.lateral_c5 = nn.Conv2d(in_channels[3], num_channels, 1, 1, padding=0, bias=True)

            self.tail_c5 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, bias=True)
            self.tail_c4 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, bias=True)
            self.tail_c3 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, bias=True)
            self.tail_c2 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, bias=True)
        else:
            self.lateral_c3 = nn.Conv2d(in_channels[0], num_channels, 1, 1, padding=0, bias=True)
            self.lateral_c4 = nn.Conv2d(in_channels[1], num_channels, 1, 1, padding=0, bias=True)
            self.lateral_c5 = nn.Conv2d(in_channels[2], num_channels, 1, 1, padding=0, bias=True)

            self.tail_c5 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, bias=True)
            self.tail_c4 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, bias=True)
            self.tail_c3 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, bias=True)

        utils.init_xavier(self)

    def forward(self, endpoints):
        """Returns a pyramid features from bottom to up (large to small)"""
        C5 = endpoints['C5']
        C4 = endpoints['C4']
        C3 = endpoints['C3']
        no_pooling = C4.size(2) == C5.size(2)

        P5 = self.lateral_c5(C5)
        P4 = self.lateral_c4(C4) + P5 if no_pooling else \
             self.lateral_c4(C4) + F.upsample(P5, scale_factor=2, mode='nearest')
        P3 = self.lateral_c3(C3) + F.upsample(P4, scale_factor=2, mode='nearest')

        if self.with_c2:
            C2 = endpoints['C2']
            P2 = self.lateral_c2(C2) + F.upsample(P3, scale_factor=2, mode='nearest')
            P2 = self.tail_c2(P2)
            P3 = self.tail_c3(P3)
            P4 = self.tail_c4(P4)
            P5 = self.tail_c5(P5)
            return [P2, P3, P4, P5]
        else:
            P3 = self.tail_c3(P3)
            P4 = self.tail_c4(P4)
            P5 = self.tail_c5(P5)
            return [P3, P4, P5]