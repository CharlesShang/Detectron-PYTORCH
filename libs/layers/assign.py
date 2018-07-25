#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
_DEBUG = False

def assign_boxes(gt_boxes, min_k=2, max_k=5, base_size=224.0):
    """assigning boxes to layers in a pyramid according to its area
    Params
    -----
    gt_boxes: of shape (N, 5), each entry is [x1, y1, x2, y2, cls]
    strides:  the stride of each layer, like [4, 8, 16, 32]

    Returns
    -----
    layer_ids: of shape (N,), each entry is a id indicating the assigned layer id
    """
    np.seterr(all='raise')
    k0 = 4
    if gt_boxes.size > 0:
        layer_ids = np.zeros((gt_boxes.shape[0], ), dtype=np.int32)
        ws = gt_boxes[:, 2] - gt_boxes[:, 0]
        hs = gt_boxes[:, 3] - gt_boxes[:, 1]
        areas = ws * hs
        try:
            k = np.floor(k0 + np.log2(np.sqrt(areas) / base_size))
        except:
            print (gt_boxes)
            raise
        inds = np.where(k < min_k)[0]
        k[inds] = min_k
        inds = np.where(k > max_k)[0]
        k[inds] = max_k
        if _DEBUG: 
            print ("### boxes and layer ids")
            print (np.hstack((gt_boxes[:, 0:4], k[:, np.newaxis])))
        return k.astype(np.int32)

    else:
        return np.asarray([], dtype=np.int32)
