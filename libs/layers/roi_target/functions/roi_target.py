#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Function
from .._ext import roi_target


# TODO use save_for_backward instead
class RoITargetFunction(Function):
    def __init__(self, fg_overlap):
        self.fg_overlap = fg_overlap

    def forward(self, rois, roi_batch_inds, gt_boxes, gt_batch_inds):
        num_rois = rois.size(0)

        labels = rois.new(num_rois).zero_().add_(-1).long()
        deltas = torch.zeros_like(rois)
        bbwgts = torch.zeros_like(rois)
        if rois.is_cuda:
            roi_target.roi_target_forward_cuda(rois, roi_batch_inds, gt_boxes, gt_batch_inds, 0., self.fg_overlap,
                                               labels, deltas, bbwgts)
        else:
            raise NotImplementedError

        return labels, deltas, bbwgts

    def backward(self, grad_output):
        pass
