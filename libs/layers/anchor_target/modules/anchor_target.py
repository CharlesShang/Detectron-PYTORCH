#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch

from torch.nn.modules.module import Module
from ..functions.anchor_target import AnchorTargetFunction


class AnchorTarget(Module):
    def __init__(self, bg_threshold, fg_threshold,
                 ignored_threshold=0.2,
                 box_encoding='fastrcnn'):
        super(AnchorTarget, self).__init__()
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.box_encoding = box_encoding
        self.ignored_threshold = ignored_threshold

    def forward(self, anchors, gt_boxes):
        """gt_boxes should be a numpy array or a list"""
        raise ValueError('Implement on cude is buggy, dont use it')
        anchors = anchors.contiguous()
        num_anchors = anchors.size(0)
        if isinstance(gt_boxes, list):
            ret = []
            for gb in gt_boxes:
                if gb.size > 0:
                    gb = torch.from_numpy(gb).cuda()
                    labels, deltas, bbwgts, overlaps = AnchorTargetFunction(
                        self.bg_threshold, self.fg_threshold,
                        self.ignored_threshold)(anchors, gb)
                    _, inds = overlaps.max(dim=0)
                    labels[inds] = gb[:, 4].long()
                else:
                    labels = anchors.new(num_anchors).zero_().long()
                    deltas = torch.zeros_like(anchors)
                    bbwgts = torch.zeros_like(anchors)
                ret.append([labels, deltas, bbwgts, overlaps])
            labels = torch.cat([r[0] for r in ret], dim=0)
            deltas = torch.cat([r[1] for r in ret], dim=0)
            bbwgts = torch.cat([r[2] for r in ret], dim=0)
        else:
            if gt_boxes.size > 0:
                gt_boxes = torch.from_numpy(gt_boxes).cuda()
                labels, deltas, bbwgts, overlaps = AnchorTargetFunction(
                    self.bg_threshold, self.fg_threshold,
                    self.ignored_threshold)(anchors, gt_boxes)
                _, inds = overlaps.max(dim=0)
                labels[inds] = gt_boxes[:, 4].long()
            else:
                labels = anchors.new(num_anchors).zero_().long()
                deltas = torch.zeros_like(anchors)
                bbwgts = torch.zeros_like(anchors)

        return labels, deltas, bbwgts


