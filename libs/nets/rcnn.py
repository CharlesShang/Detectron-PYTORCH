#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from . import utils
from math import log


class RCNN(nn.Module):
    """apply rcnn on the cropped feature maps"""
    def __init__(self, num_channels, num_classes, feat_height, feat_width, activation='softmax'):
        super(RCNN, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.trans = nn.Sequential(
            nn.Linear(num_channels * feat_height * feat_width, 1024),
            nn.ReLU(False),
            nn.Linear(1024, 1024),
            nn.ReLU(False)
        )

        self.cls_out = nn.Linear(1024, num_classes)
        self.box_out = nn.Linear(1024, 4)
        # utils.init_msra(self)
        utils.init_guass(self.trans, 0.01)
        utils.init_guass(self.cls_out, 0.01)
        utils.init_guass(self.box_out, 0.001)

        if activation == 'sigmoid':
            self.cls_out.bias.data[:] = -2.19
        elif activation == 'softmax':
            self.cls_out.bias.data[:] = 0.
            self.cls_out.bias.data[0::num_classes] = log(9 * (num_classes - 1))
        else:
            raise ValueError('Unknow activation %s' % activation)

    def forward(self, pooled_features):
        x = pooled_features.view(pooled_features.size(0), -1)
        x = self.trans(x)
        cls = self.cls_out(x)
        box = self.box_out(x)
        return [cls, box]
