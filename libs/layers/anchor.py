#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import libs.boxes.cython_bbox as cython_bbox
import libs.configs.config as cfg
import libs.boxes.bbox_transform as bbox_transform
from libs.logs.log import LOG

_DEBUG = False


def encode(gt_boxes, all_anchors):
    """Single Shot
    Sampling

    Parameters
    ---------
    gt_boxes: an array of shape (G x 5), [x1, y1, x2, y2, class]
    all_anchors: an array of shape (h, w, A, 4),
    Returns
    --------
    labels:   Nx1 array in [-1, num_classes], negative labels are ignored
    bbox_targets: N x (4) regression targets
    bbox_inside_weights: N x (4), in {0, 1} indicating to which class is assigned.
    """

    all_anchors = all_anchors.reshape([-1, 4])
    anchors = all_anchors
    total_anchors = all_anchors.shape[0]
    bbox_flags_ = np.zeros([total_anchors], dtype=np.int32)

    if gt_boxes.size > 0:
        overlaps = cython_bbox.bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

        gt_assignment = overlaps.argmax(axis=1)  # (A)
        max_overlaps = overlaps[np.arange(total_anchors), gt_assignment]
        gt_argmax_overlaps = overlaps.argmax(axis=0)  # (G)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        labels = gt_boxes[gt_assignment, 4]
        labels[max_overlaps < cfg.rpn_bg_threshold] = 0
        labels[np.logical_and(max_overlaps < cfg.rpn_fg_threshold,
                              max_overlaps >= cfg.rpn_bg_threshold)] = -1
        bbox_flags_[max_overlaps >= 0.5] = 1

        # fg label: for each gt, hard-assign anchor with highest overlap despite its overlaps
        labels[gt_argmax_overlaps] = gt_boxes[gt_assignment[gt_argmax_overlaps], 4]
        # bbox_flags_[gt_argmax_overlaps] = 1

        # if clobber positive: there may exist some positive objs (jaccard overlap < bg_th) that are not assigned to any anchors
        if cfg.rpn_clobber_positives:
            labels[max_overlaps < cfg.rpn_bg_threshold] = 0
        bbox_flags_[labels >= 1] = 1

        if _DEBUG:
            min_ov = np.min(gt_max_overlaps)
            max_ov = np.max(gt_max_overlaps)
            mean_ov = np.mean(gt_max_overlaps)
            if min_ov < cfg.rpn_bg_threshold:
                LOG('ANCHORSS: overlaps: (min %.3f mean:%.3f max:%.3f)' % (min_ov, mean_ov, max_ov))
                worst = gt_boxes[np.argmin(gt_max_overlaps)]
                anc = anchors[gt_argmax_overlaps[np.argmin(gt_max_overlaps)], :]
                LOG('ANCHORSS: worst overlap:%.3f, box:(%.1f, %.1f, %.1f, %.1f %d), anchor:(%.1f, %.1f, %.1f, %.1f)'
                    % (min_ov, worst[0], worst[1], worst[2], worst[3], worst[4],
                       anc[0], anc[1], anc[2], anc[3]))

        ## handle ignored regions (the gt_class of crowd boxes is set to -1)
        ignored_inds = np.where(gt_boxes[:, -1] < 0)[0]
        if ignored_inds.size > 0:
            ignored_areas = gt_boxes[ignored_inds, :]
            # intersec shape is D x A
            intersecs = cython_bbox.bbox_intersections(
                np.ascontiguousarray(ignored_areas, dtype=np.float),
                np.ascontiguousarray(anchors, dtype=np.float)
            )
            intersecs_ = intersecs.sum(axis=0)  # A x 1
            labels[intersecs_ > cfg.ignored_area_intersection_fraction] = -1
            bbox_flags_[intersecs_ > cfg.ignored_area_intersection_fraction] = 0

    else:
        # if there is no gt
        labels = np.zeros([total_anchors], dtype=np.float32)

    label_weights = np.zeros((total_anchors,), dtype=np.float32)

    if cfg.rpn_sample_strategy == 'traditional':
        """subsample positive labels if there are too many, inherited from fastrcnn"""
        num_fg = int(cfg.rpn_fg_fraction * cfg.rpn_batch_size)
        fg_inds = np.where(labels >= 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
        else:
            num_fg = len(fg_inds)
        # subsample negative labels if there are too many
        num_bg = max(min(cfg.rpn_batch_size - num_fg, num_fg * 5), 128)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

    elif cfg.rpn_sample_strategy == 'simple':
        """using label_weights to balance example losses"""
        fg_inds = np.where(labels >= 1)[0]
        num_fg = len(fg_inds)
        label_weights[fg_inds] = 1.0
        bg_inds = np.where(labels == 0)[0]
        num_bg = len(bg_inds)
        label_weights[bg_inds] = 3 * max(num_fg, 1.0) / max(max(num_bg, num_fg), 1.0)

    elif cfg.rpn_sample_strategy == 'advanced':
        """no implemented yet"""
        # deal with ignored lables?
    else:
        raise ValueError('RPN sample strategy %s has not been implemented yet' % cfg.rpn_sample_strategy)

    # if True: # person only
    #     nonperson_inds = np.where(np.logical_and(labels != 1, labels != -1))[0]
    #     labels[nonperson_inds] = 0
    #     label_weights[nonperson_inds] = 0
    #     kept_inds = np.random.choice(nonperson_inds, size=(1000), replace=False)
    #     label_weights[kept_inds] = 0.02

    bbox_targets = np.zeros((total_anchors, 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets = _compute_targets(anchors, gt_boxes[gt_assignment, :])
    bbox_inside_weights = np.zeros((total_anchors, 4), dtype=np.float32)
    # bbox_inside_weights[labels >= 1, :] = np.asarray(cfg.bbweights, dtype=np.float32)
    bbox_inside_weights[bbox_flags_ == 1, :] = np.asarray(cfg.bbweights, dtype=np.float32)

    labels = labels.reshape((-1,))
    bbox_targets = bbox_targets.reshape((-1, 4))
    bbox_inside_weights = bbox_inside_weights.reshape((-1, 4))

    return labels.astype(np.float32), label_weights, bbox_targets.astype(np.float32), bbox_inside_weights.astype(
        np.float32)


def decode(boxes, scores, all_anchors, ih, iw, num_classes=None):
    """Decode outputs into boxes
    Parameters
    ---------
    boxes: an array of shape (1, h, w, Ax4)
    scores: an array of shape (1, h, w, Ax2),
    all_anchors: an array of shape (1, h, w, Ax4), [x1, y1, x2, y2]

    Returns
    --------
    final_boxes: of shape (R x 4)
    classes: of shape (R) in {0,1,2,3... K-1}
    scores: of shape (R, K) in [0 ~ 1]
    """
    num_classes = cfg.num_classes if num_classes is None else num_classes
    all_anchors = all_anchors.reshape((-1, 4))
    boxes = boxes.reshape((-1, 4))
    scores = scores.reshape((-1, num_classes))
    assert scores.shape[0] == boxes.shape[0] == all_anchors.shape[0], \
        'Anchor layer shape error %d vs %d vs %d' % (scores.shape[0], boxes.shape[0], all_anchors.reshape[0])
    if cfg.rpn_box_encoding == 'fastrcnn':
        boxes = bbox_transform.bbox_transform_inv(all_anchors, boxes)
    elif cfg.rpn_box_encoding == 'linear':
        boxes = bbox_transform.bbox_transform_inv_linear(all_anchors, boxes)
    classes = np.argmax(scores, axis=1)
    final_boxes = boxes
    final_boxes = bbox_transform.clip_boxes(final_boxes, (ih, iw))
    classes = classes.astype(np.int32)
    return final_boxes, classes, scores


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    if cfg.rpn_box_encoding == 'linear':
        return bbox_transform.bbox_transform_linear(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
    if cfg.rpn_box_encoding == 'fastrcnn':
        return bbox_transform.bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
