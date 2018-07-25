#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch

from libs.layers.anchor_target.modules.anchor_target import AnchorTarget
import libs.boxes.cython_bbox as cython_bbox


if __name__ == '__main__':

    anchors = [
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
    gt_boxes = [
        [121, 122, 144, 155, 2],
        [1, 1, 3, 3, 1],
        [0, 0, 10, 10, -1]
    ]
    anchors = np.asarray(anchors, dtype=np.float32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)

    overlaps = cython_bbox.bbox_overlaps(
        np.ascontiguousarray(anchors[:, :4], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

    gt_assignment = overlaps.argmax(axis=1)  # (R)
    max_overlaps = overlaps[np.arange(anchors.shape[0], ), gt_assignment]

    anchors = torch.from_numpy(anchors).cuda()

    anchor_target = AnchorTarget(0.3, 0.6)
    labels, deltas, bbwght = anchor_target(anchors, gt_boxes)

    labels.cpu()
    print(labels.size(), deltas.size(), bbwght.size())
    print(labels, deltas, bbwght)
    print(overlaps)
