#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import libs.boxes.cython_bbox as cython_bbox

BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.

def sample_distored_bounding_box(im, bboxes, labels, inst_masks, mask,
                                 min_obj_cover=0.5, min_img_cover=0.3,
                                 min_aspect_ratio=0.667, max_aspect_ratio=1.667):

    if bboxes.size == 0:
        return im, bboxes, labels, inst_masks, mask

    im_height, im_width, = im.shape[:2]
    boxes = _generate_random_boxes(im_height, im_width,
                                   min_aspect_ratio=min_aspect_ratio, max_aspect_ratio=max_aspect_ratio,
                                   min_img_cover=min_img_cover)
    box = _choose_a_valid_box(boxes, bboxes, min_obj_cover=min_obj_cover)
    if box is None:
        return im, bboxes, labels, inst_masks, mask

    im, bboxes, labels, inst_masks, mask = \
        _compute_and_filter_valid_annotations(box, im, bboxes, labels, inst_masks, mask)

    return im, bboxes, labels, inst_masks, mask

def _generate_random_boxes(im_height, im_width,
                           min_img_cover=0.33, max_img_cover=1.0,
                           min_aspect_ratio=0.667, max_aspect_ratio=1.667,
                           num_attempts=500):
    max_height = min(im_width / max_aspect_ratio, im_height * max_img_cover)
    max_height = max(im_height * min_img_cover + 1, max_height) # in case max_height is less than im_height * min_img_cover
    heights = np.random.randint(int(im_height * min_img_cover), int(max_height), [num_attempts]).astype(np.float32)
    aspect_ratios = np.random.rand(num_attempts) * (max_aspect_ratio - min_aspect_ratio) + min_aspect_ratio
    widths = heights * aspect_ratios
    widths[widths > im_width] = im_width - 1.
    xs = np.random.rand(num_attempts) * (im_width - widths)
    ys = np.random.rand(num_attempts) * (im_height - heights)

    boxes = np.vstack((xs, ys, widths + xs - 1, heights + ys - 1)).transpose()
    return np.floor(boxes)



def _choose_a_valid_box(boxes, bboxes, min_obj_cover=0.5, default_box=None):
    # intersec shape is D x A
    intersecs = cython_bbox.bbox_intersections(
        np.ascontiguousarray(boxes, dtype=np.float),
        np.ascontiguousarray(bboxes, dtype=np.float))
    min_obj_covers = intersecs.min(axis=1)
    # excluded = np.where(np.logical_and(min_obj_covers < min_obj_cover, min_obj_covers > 0.001))[0]
    # inds = np.setdiff1d(np.arange(boxes.shape[0]), excluded)
    inds = np.where(min_obj_covers > min_obj_cover)[0]
    if inds.size == 0:
        return default_box
    else:
        ind = np.random.choice(inds, 1)
        return boxes[ind[0], :]

def _filter_invalid_boxes_with_size(boxes, min_size=4):
    """drop small boxes"""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    ignored_inds = np.where(np.logical_or(ws < min_size, hs < min_size))[0]
    return ignored_inds

def _filter_invalid_boxes_with_cover(boxes, ori_boxes, min_ratio=0.5):
    """drop small boxes"""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    areas = ws * hs

    ws = ori_boxes[:, 2] - ori_boxes[:, 0] + 1
    hs = ori_boxes[:, 3] - ori_boxes[:, 1] + 1
    ori_areas = ws * hs
    ratios = areas.astype(np.float32) / ori_areas.astype(np.float32)
    ignored_inds = np.where(ratios < min_ratio)[0]

    return ignored_inds

def _compute_and_filter_valid_annotations(box, im, bboxes, labels, inst_masks, mask):
    """compute new annotations and filter valid"""
    h, w = box[3] - box[1], box[2] - box[0]
    new_boxes = bboxes.copy()
    new_boxes[:, 0] -= box[0]
    new_boxes[:, 1] -= box[1]
    new_boxes[:, 2] -= box[0]
    new_boxes[:, 3] -= box[1]
    x1s = new_boxes[:, 0]
    y1s = new_boxes[:, 1]
    x2s = new_boxes[:, 2]
    y2s = new_boxes[:, 3]
    x1s[x1s < 0] = 0
    y1s[y1s < 0] = 0
    x2s[x2s >= w] = w - 1
    y2s[y2s >= h] = h - 1
    new_boxes[:, 2] = x2s
    new_boxes[:, 3] = y2s
    new_boxes[:, 0] = x1s
    new_boxes[:, 1] = y1s
    new_boxes = np.floor(new_boxes)

    ignored_inds1 = _filter_invalid_boxes_with_size(new_boxes)
    ignored_inds2 = _filter_invalid_boxes_with_cover(new_boxes, bboxes)
    keep_inds = np.setdiff1d(np.arange(new_boxes.shape[0]), ignored_inds1)
    # keep_inds = np.setdiff1d(keep_inds, ignored_inds2)
    labels[ignored_inds2] = -1 # set to ignore

    b = np.floor(box).astype(np.int32)
    # assert b[3]-b[1] > 10 and b[2] - b[0] > 10
    im = im[b[1]:b[3], b[0]:b[2]]
    inst_masks = inst_masks[:, b[1]:b[3], b[0]:b[2]]
    mask = mask[b[1]:b[3], b[0]:b[2]]

    if keep_inds.size == 0:
        h, w = im.shape[:2]
        return im, np.zeros([0, 4], dtype=bboxes.dtype), np.zeros([0], dtype=labels.dtype), \
               np.zeros([0, h, w], dtype=inst_masks.dtype), np.zeros([h, w], dtype=mask.dtype),

    return im, new_boxes[keep_inds], labels[keep_inds], inst_masks[keep_inds], mask