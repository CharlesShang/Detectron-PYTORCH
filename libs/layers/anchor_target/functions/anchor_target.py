#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Function
from .._ext import anchor_target


# TODO use save_for_backward instead
class AnchorTargetFunction(Function):
    def __init__(self, bg_threshold, fg_threshold,
                 ignored_threshold=0.2):
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.ignored_threshold = ignored_threshold

    def forward(self, anchors, gt_boxes):
        num_anchors = anchors.size(0)
        num_gts = gt_boxes.size(0)

        labels = anchors.new(num_anchors).zero_().add_(-1).long()
        deltas = torch.zeros_like(anchors)
        bbwgts = torch.zeros_like(anchors)
        overlaps = anchors.new(num_anchors, num_gts).zero_()
        if anchors.is_cuda:
            anchor_target.anchor_target_forward_cuda(anchors, gt_boxes,
                                                     self.bg_threshold, self.fg_threshold, self.ignored_threshold,
                                                     labels, deltas, bbwgts, overlaps)
        else:
            raise NotImplementedError

        return labels, deltas, bbwgts, overlaps

    def backward(self, grad_output):
        pass
