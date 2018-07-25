#!/usr/bin/env python
# coding=utf-8
# This file is copied from torchvision.models
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from . import utils


class Segnet(nn.Module):

    def __init__(self, in_channels, num_classes, stride, num_repeat=3, num_channels=256, withBN=False):
        super(Segnet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.stride = stride
        self.repeat = num_repeat
        self.convs = []

        conv_combo = utils.Conv2d_BatchNorm if withBN else utils.Conv2d
        self.convs = nn.Sequential(
            conv_combo(in_channels, self.num_channels, 3, same_padding=True),
            conv_combo(num_channels, self.num_channels, 3, same_padding=True),
            conv_combo(num_channels, self.num_channels, 3, same_padding=True),
        )

        self.conv_segm_out = nn.Conv2d(self.num_channels, num_classes, kernel_size=3, padding=1, bias=True)
        utils.init_new_layers(self.conv_segm_out)


    def forward(self, x):
        x = self.convs(x)
        x = self.conv_segm_out(x)
        return x