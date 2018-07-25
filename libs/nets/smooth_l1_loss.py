#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def smooth_l1_loss(bbox_preds, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=3.0):
    """as caffe implementation, the bbox_preds and bbox_targets should be a 2-dim torch Var
    [[x1, y1, x2, y2],
     [x1, y1, x2, y2],].
    """
    sigma2 = sigma ** 2
    diff = bbox_preds - bbox_targets
    diff = diff * bbox_inside_weights
    diff_abs = torch.abs(diff)
    smooth_l1_sign = (diff_abs < 1. / sigma2).detach().float()
    box_loss = torch.pow(diff, 2) * (sigma2 * 0.5) * smooth_l1_sign \
                + (diff_abs - (0.5 / sigma2)) * (1. - smooth_l1_sign)
    box_loss = box_loss * bbox_outside_weights

    return box_loss.sum(1).sum()