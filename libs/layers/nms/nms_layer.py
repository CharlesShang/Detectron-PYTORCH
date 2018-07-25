#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Function
import torch.nn as nn
from ._ext import nms

class NMSFunction(Function):

    def __init__(self, overlap_threshold):
        super(NMSFunction, self).__init__()
        self.overlap_threshold = overlap_threshold

    def forward(self, boxes, scores):

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        dets = torch.cat((boxes, scores), dim=0)
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)

        if not dets.is_cuda:
            nms.cpu_nms(keep, num_out, dets, order, areas, self.overlap_threshold)
            return keep[:num_out[0]]

        else:
            dets = dets[order].contiguous()
            nms.gpu_nms(keep, num_out, dets, self.overlap_threshold)

            return order[keep[:num_out[0]].cuda()].contiguous()

    def backward(self, grad_top):
        raise ValueError('backward on NMSFunction should never be called')


class NMSLayer(nn.Module):
    def __init__(self, overlap_threshold):
        super(NMSLayer, self).__init__()
        self.overlap_threshold = overlap_threshold


    def forward(self, boxes, scores):
        keeps = NMSFunction(self.overlap_threshold)(boxes, scores)
        return keeps

    def backward(self, grad_top):
        raise ValueError('backward on NMSLayer should never be called')

