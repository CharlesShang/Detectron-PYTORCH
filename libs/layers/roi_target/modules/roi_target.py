#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch

from torch.nn.modules.module import Module
from ..functions.roi_target import RoITargetFunction


class RoITarget(Module):
    def __init__(self, fg_overlap, box_encoding='fastrcnn'):
        super(RoITarget, self).__init__()
        self.fg_overlap = fg_overlap
        self.box_encoding = box_encoding

    def forward(self, rois, roi_batch_inds, gt_boxes, gt_batch_inds):
        # gt_boxes = gt_boxes[:, :4]
        # classes = gt_boxes[:, 4].long()
        rois = rois[:, :4]
        gt_boxes = gt_boxes.contiguous()
        gt_batch_inds = gt_batch_inds.contiguous()
        rois = rois.contiguous()
        labels, deltas, bbwgts = RoITargetFunction(self.fg_overlap)(rois, roi_batch_inds, gt_boxes, gt_batch_inds)

        return labels, deltas, bbwgts
