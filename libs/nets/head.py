#!/usr/bin/env python
# coding=utf-8
# This file is copied from torchvision.models
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from . import utils
from math import log

class RPNHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors, num_channels=256, activation='sigmoid'):
        super(RPNHead, self).__init__()

        conv_combo = utils.Conv2d
        self.conv = conv_combo(in_channels, num_channels, 3, same_padding=True)
        self.conv_cls_out = nn.Conv2d(num_channels, num_anchors * 2, kernel_size=1, padding=0, bias=True)
        self.conv_box_out = nn.Conv2d(num_channels, num_anchors * 4, kernel_size=1, padding=0, bias=True)

        # utils.init_xavier(self)
        utils.init_guass(self.conv, 0.01)
        utils.init_guass(self.conv_cls_out, std=0.01)
        utils.init_guass(self.conv_box_out, std=0.001)

        # set background prior larger than foreground to prevent network blowup in the early iterations
        self.conv_cls_out.bias.data.zero_()
        if activation == 'sigmoid':
            self.conv_cls_out.bias.data[:] = -4.595
        else:
            # self.conv_cls_out.bias.data[:] = -2.
            # self.conv_cls_out.bias.data[0::num_classes] = log(0.54 * (num_classes - 1))
            self.conv_cls_out.bias.data[:] = 0.
            self.conv_cls_out.bias.data[0::num_classes] = log(99 * (num_classes - 1))

    def forward(self, x):
        rpn_logits = self.conv(x)
        return [self.conv_cls_out(rpn_logits), self.conv_box_out(rpn_logits)]


class RetinaHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors, num_channels=256, activation='sigmoid'):
        super(RetinaHead, self).__init__()

        conv_combo = utils.Conv2d
        self.conv_box = nn.Sequential(
            conv_combo(in_channels, num_channels, 3, same_padding=True),
            conv_combo(num_channels, num_channels, 3, same_padding=True),
            conv_combo(num_channels, num_channels, 3, same_padding=True),
            conv_combo(num_channels, num_channels, 3, same_padding=True),
        )
        self.conv_cls = nn.Sequential(
            conv_combo(in_channels, num_channels, 3, same_padding=True),
            conv_combo(num_channels, num_channels, 3, same_padding=True),
            conv_combo(num_channels, num_channels, 3, same_padding=True),
            conv_combo(num_channels, num_channels, 3, same_padding=True),
        )

        self.conv_cls_out = nn.Conv2d(num_channels, num_anchors * num_classes,
                                      kernel_size=3, padding=1, bias=True)
        self.conv_box_out = nn.Conv2d(num_channels, num_anchors * 4,
                                      kernel_size=3, padding=1, bias=True)

        utils.init_xavier(self)
        # utils.init_guass(self, 0.01)
        utils.init_guass(self.conv_box_out, 0.001)
        utils.init_guass(self.conv_cls_out, 0.01)
        # utils.init_conv_weight(self.conv_cls_out, std=0.01)
        # utils.init_conv_weight(self.conv_box_out, std=0.001)

        # set background prior larger than foreground to prevent network blowup in the early iterations
        self.conv_cls_out.bias.data.zero_()
        if activation == 'sigmoid':
            self.conv_cls_out.bias.data[:] = -4.595
        else:
            self.conv_cls_out.bias.data[:] = 0.
            self.conv_cls_out.bias.data[0::num_classes] = log(99 * (num_classes - 1))
            # self.conv_cls_out.bias.data[:] = -2.
            # self.conv_cls_out.bias.data[0::num_classes] = log(0.54 * (num_classes - 1))

    def forward(self, x):
        rpn_logits = self.conv_cls(x)
        rpn_box = self.conv_box(x)
        return [self.conv_cls_out(rpn_logits), self.conv_box_out(rpn_box)]
