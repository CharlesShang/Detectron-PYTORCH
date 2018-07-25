# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import warnings

def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    # assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        # 'Invalid boxes found: {} {}'. \
            # format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths / 0.1
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights / 0.1
    targets_dw = np.log(gt_widths / ex_widths) / 0.2
    targets_dh = np.log(gt_heights / ex_heights) / 0.2

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4] * 0.1
    dy = deltas[:, 1::4] * 0.1
    dw = deltas[:, 2::4] * 0.2
    dh = deltas[:, 3::4] * 0.2

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    # pred_w = np.exp(dw) * widths[:, np.newaxis]
    # pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_w = np.exp(dw + np.log(widths[:, np.newaxis]))
    pred_h = np.exp(dh + np.log(heights[:, np.newaxis]))

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def bbox_transform_linear(anchors, gt_boxes, alpha=10.0):
    """
    computes the distance from ground-truth boxes to the given boxes, normalized by their size
    :param anchors: n * 4 numpy array, given boxes
    :param gt_boxes: n * 4 numpy array, ground-truth boxes
    :return: targets: n * 4 numpy array, ground-truth boxes
    """
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    # sizes = np.sqrt(widths * heights)

    targets = gt_boxes - anchors

    # targets[:, 0] = targets[:, 0] / sizes * alpha
    # targets[:, 1] = targets[:, 1] / sizes * alpha
    # targets[:, 2] = targets[:, 2] / sizes * alpha
    # targets[:, 3] = targets[:, 3] / sizes * alpha

    targets[:, 0] = targets[:, 0] / widths * alpha
    targets[:, 1] = targets[:, 1] / heights * alpha
    targets[:, 2] = targets[:, 2] / widths * alpha
    targets[:, 3] = targets[:, 3] / heights * alpha

    return targets

def bbox_transform_inv_linear(anchors, deltas, alpha=10.0):
    if anchors.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    # sizes = np.sqrt(widths * heights)

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    # try:
    #     sizes = np.sqrt(widths * heights)
    # except:
    #     print (np.min(widths), np.max(heights))

    pred_boxes = anchors.copy()
    # pred_boxes[:, 0] = deltas[:, 0] * sizes / alpha + pred_boxes[:, 0]
    # pred_boxes[:, 1] = deltas[:, 1] * sizes / alpha + pred_boxes[:, 1]
    # pred_boxes[:, 2] = deltas[:, 2] * sizes / alpha + pred_boxes[:, 2]
    # pred_boxes[:, 3] = deltas[:, 3] * sizes / alpha + pred_boxes[:, 3]

    pred_boxes[:, 0] = deltas[:, 0] * widths / alpha + pred_boxes[:, 0]
    pred_boxes[:, 1] = deltas[:, 1] * heights / alpha + pred_boxes[:, 1]
    pred_boxes[:, 2] = deltas[:, 2] * widths / alpha + pred_boxes[:, 2]
    pred_boxes[:, 3] = deltas[:, 3] * heights / alpha + pred_boxes[:, 3]

    return pred_boxes
