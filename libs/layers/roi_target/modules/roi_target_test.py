#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
from torch.autograd import Variable
from libs.layers.roi_target.modules.roi_target import RoITarget
import libs.boxes.cython_bbox as cython_bbox


if __name__ == '__main__':

    rois = [
        [120, 120, 140, 150],
        [95, 100, 120, 130],
        [120, 120, 125, 125],
        [40, 50, 50, 60],
        [95, 100, 120, 130],
        [0, 0, 9, 9],
        [1, 1, 8, 8],
        [10, 10, 20, 20],
        [16, 10, 26, 20],
        [70, 60, 80, 90],
    ]
    roi_batch_inds = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    gt_boxes = [
        [121, 120, 140, 150, 0],
        [100, 100, 120, 130, 0],
        [121, 120, 140, 150, -1],
        [100, 100, 120, 130, -1],
        [1, 1, 8, 8, 3],
        [13, 10, 23, 20, 4],
    ]
    gt_batch_inds = [0, 0, 1, 1, 1, 1]
    rois = np.asarray(rois, dtype=np.float32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    gt_batch_inds = np.asarray(gt_batch_inds, dtype=np.float32)
    roi_batch_inds = np.asarray(roi_batch_inds, dtype=np.float32)

    overlaps = cython_bbox.bbox_overlaps(
        np.ascontiguousarray(rois[:, :4], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

    gt_assignment = overlaps.argmax(axis=1)  # (R)
    max_overlaps = overlaps[np.arange(rois.shape[0], ), gt_assignment]

    rois = Variable(torch.from_numpy(rois)).cuda()
    gt_boxes = Variable(torch.from_numpy(gt_boxes)).cuda()
    gt_batch_inds = Variable(torch.from_numpy(gt_batch_inds)).cuda().long()
    roi_batch_inds = Variable(torch.from_numpy(roi_batch_inds)).cuda().long()

    roi_target = RoITarget(0.55, box_encoding='fastrcnn')
    labels, deltas, bbwght = roi_target(rois, roi_batch_inds, gt_boxes, gt_batch_inds)

    labels.cpu()
    print(labels.size(), deltas.size(), bbwght.size())
    print(labels, deltas, bbwght)
