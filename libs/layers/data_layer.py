#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import libs.configs.config as cfg
from libs.preprocessings.fixed_size import preprocess_train, preprocess_test, \
    preprocess_train_keep_aspect_ratio, preprocess_test_keep_aspect_ratio, preprocess_eval_keep_aspect_ratio
from libs.boxes.anchor import anchors_plane, anchor_pyramid

from . import anchor


def data_layer(img_name, bboxes, classes, masks, mask, is_training, ANCHORS=[]):
    """ Returns the learning labels
    1. resize image, boxes, masks, mask
    2. data augmentation
    3. build learning labels
    """
    im = cv2.imread(img_name).astype(np.float32)
    if im.size == im.shape[0] * im.shape[1]:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = im.astype(np.float32)

    strides = cfg.strides

    if is_training:
        im, bboxes, classes, masks, mask, ori_im = \
            preprocess_train(im, bboxes, classes, masks, mask, cfg.input_size, cfg.min_size,
                             use_augment=cfg.use_augment, training_scale=cfg.training_scale)
        gt_boxes = np.hstack((bboxes, classes[:, np.newaxis]))
        # layer_ids = assign.assign_boxes(gt_boxes, min_k=int(np.log2(strides[0])), max_k=int(np.log2(strides[-1])),
        #                                 base_size=cfg.base_size)
    else:
        im, ori_im = \
            preprocess_test(im, cfg.input_size)
        masks, mask = [], []

    ih, iw = im.shape[0:2]

    ANNOTATIONS = []
    # if is_training:
    ANNOTATIONS = [bboxes, classes]

    if len(ANCHORS) == 0:
        for i, stride in enumerate(strides):

            height, width = int(ih / stride), int(iw / stride)
            scales = cfg.anchor_scales[i] if isinstance(cfg.anchor_scales[i], list) else cfg.anchor_scales
            all_anchors = anchors_plane(height, width, stride,
                                        scales=scales,
                                        ratios=cfg.anchor_ratios,
                                        base=cfg.anchor_base)
            ANCHORS.append(all_anchors)

    all_anchors = []
    for i in range(len(ANCHORS)):
        all_anchors.append(ANCHORS[i].reshape((-1, 4)))
    all_anchors = np.vstack(all_anchors)

    # building learning labels
    TARGETS = []
    if is_training:
        labels, label_weights, bbox_targets, bbox_inside_weights = \
            anchor.encode(gt_boxes, all_anchors)
        TARGETS = [labels, label_weights, bbox_targets, bbox_inside_weights] # flat (N, ), (N, 4), (N, 4)

    # if _DEBUG:
    #     np.set_printoptions(precision=3)
    #     bb = bbox_targets[labels > 0, :]
    #     mean = np.abs(bb).mean(0)
    #     max = np.abs(bb).max()
    #     s = bbox_targets[labels > 0, :].std()

    return im, TARGETS, masks, mask, ori_im, ANNOTATIONS


def data_layer_keep_aspect_ratio(img_name, bboxes, classes, inst_masks, mask, is_training):
    """ Returns the learning labels
    1. resize image, boxes, masks, mask
    2. data augmentation
    """
    im = cv2.imread(img_name).astype(np.float32)
    if im.size == im.shape[0] * im.shape[1]:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = im.astype(np.float32)
    annots = {}

    if is_training:
        im, bboxes, classes, inst_masks, mask, im_scale = \
            preprocess_train_keep_aspect_ratio(im, bboxes, classes, inst_masks, mask,
                                               min_side=cfg.min_side, max_side=cfg.max_side,
                                               canvas_height=cfg.canvas_height, canvas_width=cfg.canvas_width,
                                               use_augment=cfg.use_augment, training_scale=cfg.training_scale)

        anchors = anchor_pyramid(cfg.strides, cfg.canvas_height, cfg.canvas_width,
                                 cfg.anchor_scales, cfg.anchor_ratios, cfg.anchor_base, cfg.anchor_shift)
        rpn_labels, _, rpn_bbtargets, rpn_bbwghts = \
            anchor.encode(np.hstack((bboxes, classes[:, np.newaxis])), anchors)
        annots['bboxes'] = bboxes
        annots['classes'] = classes
        annots['gt_boxes'] = np.hstack((bboxes, classes[:, np.newaxis]))
        annots['inst_masks'] = inst_masks
        annots['mask'] = mask
        annots['rpn_targets'] = [rpn_labels, rpn_bbtargets, rpn_bbwghts]
    elif bboxes is not None:
        im, bboxes, classes, inst_masks, mask, im_scale = \
            preprocess_eval_keep_aspect_ratio(im, bboxes, classes, inst_masks, mask,
                                              min_side=cfg.min_side, max_side=cfg.max_side)
        annots['bboxes'] = bboxes
        annots['classes'] = classes
        annots['gt_boxes'] = np.hstack((bboxes, classes[:, np.newaxis]))
        annots['inst_masks'] = inst_masks
        annots['mask'] = mask
        annots['rpn_targets'] = []
    else:
        im, im_scale = preprocess_test_keep_aspect_ratio(im, min_side=cfg.min_side, max_side=cfg.max_side)

    return im, im_scale, annots


def data_layer_keep_aspect_ratio_batch(raw_batch, is_training, canvas_width=None, canvas_height=None):
    """ Returns the learning labels
    build learning labels
    """
    if canvas_width is None:
        heights = np.asarray([d[0].shape[0] for d in raw_batch], dtype=np.int)
        widths = np.asarray([d[0].shape[1] for d in raw_batch], dtype=np.int)
        max_height = heights.max()
        max_width = widths.max()
        max_stride = np.asarray(cfg.strides, dtype=np.int).max()
        canvas_width = int(np.ceil(max_width / max_stride) * max_stride)
        canvas_height = int(np.ceil(max_height / max_stride) * max_stride)

    anchors = anchor_pyramid(cfg.strides, canvas_height, canvas_width,
                             cfg.anchor_scales, cfg.anchor_ratios, cfg.anchor_base, cfg.anchor_shift)
    anchors = anchors.astype(np.float32)

    im_batch = []
    labels_batch = []
    labels_weights_batch = []
    bbox_targets_batch = []
    bbox_inside_weights_batch = []
    mask_batch = []
    inst_masks_batch = []
    im_scale_batch = []

    for d in raw_batch:
        im, im_scale = d[:2]
        ih, iw, c = im.shape
        canvas_im = np.zeros((canvas_height, canvas_width, 3), dtype=im.dtype)
        canvas_im[:ih, :iw, :] = im
        canvas_im = np.transpose(canvas_im, [2, 0, 1])
        im_batch.append(canvas_im)
        im_scale_batch.append(im_scale)

    if not is_training:
        return im_batch, im_scale_batch, anchors, {}, inst_masks_batch, mask_batch

    for d in raw_batch:
        im, im_scale, annots, img_id = d[:4]
        ih, iw, c = im.shape

        gt_boxes = annots['gt_boxes']
        labels, label_weights, bbox_targets, bbox_inside_weights = \
            anchor.encode(gt_boxes, anchors)

        labels_batch.append(labels)
        labels_weights_batch.append(label_weights)
        bbox_targets_batch.append(bbox_targets)
        bbox_inside_weights_batch.append(bbox_inside_weights)

        mask = annots['mask']
        canvas_mask = np.zeros((canvas_height, canvas_width), dtype=mask.dtype)
        canvas_mask[:ih, :iw] = mask
        if annots['inst_masks'].size > 0:
            inst_masks = annots['inst_masks']
            canvas_inst_masks = np.zeros((inst_masks.shape[0], canvas_height, canvas_width), dtype=inst_masks.dtype)
            canvas_inst_masks[:, :ih, :iw] = inst_masks
        else:
            canvas_inst_masks = annots['inst_masks']
        inst_masks_batch.append(canvas_inst_masks)
        mask_batch.append(canvas_mask)

    rpn_targets = {'labels_batch': labels_batch,
                   'labels_weights_batch': labels_weights_batch,
                   'bbox_targets_batch': bbox_targets_batch,
                   'bbox_inside_weights_batch': bbox_inside_weights_batch
                   }

    return im_batch, im_scale_batch, anchors, rpn_targets, inst_masks_batch, mask_batch


# def data_layer_keep_aspect_ratio_batch(raw_batch, is_training):
#     """ Returns the learning labels
#     build learning labels
#     """
#     heights = np.asarray([d[0].shape[0] for d in raw_batch], dtype=np.int)
#     widths = np.asarray([d[0].shape[1] for d in raw_batch], dtype=np.int)
#     max_height = heights.max()
#     max_width = widths.max()
#     max_stride = np.asarray(cfg.strides, dtype=np.int).max()
#     canvas_width = int(np.ceil(max_width / max_stride) * max_stride)
#     canvas_height = int(np.ceil(max_height / max_stride) * max_stride)
#
#     anchors = anchor_pyramid(cfg.strides, canvas_height, canvas_width,
#                              cfg.anchor_scales, cfg.anchor_ratios, cfg.anchor_base)
#     anchors = anchors.astype(np.float32)
#
#     im_batch = []
#     im_scale_batch = []
#
#     for d in raw_batch:
#         im, im_scale = d[:2]
#         ih, iw, c = im.shape
#         canvas_im = np.zeros((canvas_height, canvas_width, 3), dtype=im.dtype)
#         canvas_im[:ih, :iw, :] = im
#         canvas_im = np.transpose(canvas_im, [2, 0, 1]) # 3, h, w
#         im_batch.append(canvas_im)
#         im_scale_batch.append(im_scale)
#
#     return im_batch, im_scale_batch, anchors


def compute_rpn_targets_in_batch(gt_boxes_list, anchors):
    batch_size = len(gt_boxes_list)
    anchors_list = anchors if isinstance(anchors, list) else [anchors] * batch_size
    target_list = []
    for gt_boxes, anchors in zip(gt_boxes_list, anchors_list):
        labels, label_weights, bbox_targets, bbox_inside_weights = \
            anchor.encode(gt_boxes, anchors)
        target_list.append([labels, label_weights, bbox_targets, bbox_inside_weights])

    # return [t[0] for t in target_list], [t[1] for t in target_list], \
    #        [t[2] for t in target_list], [t[3] for t in target_list]
    return [np.stack([t[i] for t in target_list]) for i in range(4)]
