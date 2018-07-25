#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from ..functions.roi_align import RoIAlignFunction


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)

    def forward(self, features, rois, spatial_scale):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
                                spatial_scale)(features, rois)


class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)

    def forward(self, features, rois, spatial_scale):
        x = RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                             spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)

    def forward(self, features, rois, spatial_scale):
        x = RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                             spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)
