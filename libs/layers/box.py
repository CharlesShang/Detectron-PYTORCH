#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import libs.boxes.cython_bbox as cython_bbox
from .nms._ext import nms


def matching_box(boxes, image_inds, gt_boxes_list, bg_overlap_threshold=0.5, fg_overlap_threshold=0.6):
    """gt_boxes_list is a list of np.ndarray, batch_inds specify the image a boxes belongs"""
    if boxes.is_cuda:
        boxes_np = boxes.data.cpu().numpy()
    else:
        boxes_np = boxes.data.numpy()

    if image_inds.is_cuda:
        image_inds_np = image_inds.cpu().numpy()
    else:
        image_inds_np = image_inds.numpy()

    num_boxes = boxes_np.shape[0]
    assert num_boxes == image_inds_np.size
    match_labels = []
    match_inds = []
    match_boxes = []

    for i, gt_boxes in enumerate(gt_boxes_list):
        boxes_im = boxes_np[image_inds_np == i]
        num_boxes_im = boxes_im.shape[0]
        match = np.zeros((boxes_im.shape[0],), dtype=np.int32) - 1
        labels = np.zeros((boxes_im.shape[0],), dtype=np.int64)
        match_box = np.zeros((boxes_im.shape[0], 4), dtype=np.float32)
        if gt_boxes.size > 0 and boxes_im.size > 0:
            # B x G
            overlaps = cython_bbox.bbox_overlaps(
                np.ascontiguousarray(boxes_im, dtype=np.float),
                np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

            gt_assignment = overlaps.argmax(axis=1)  # (B)
            max_overlaps = overlaps[np.arange(num_boxes_im), gt_assignment]
            match[:] = gt_assignment[:]
            match[max_overlaps < bg_overlap_threshold] = -1
            match_box[:, :4] = gt_boxes[gt_assignment, :4]

            labels[:] = gt_boxes[gt_assignment, 4]
            # labels[max_overlaps < bg_overlap_threshold] = 0
            # labels[np.logical_and(max_overlaps > bg_overlap_threshold,
            #                       max_overlaps < fg_overlap_threshold)] = -1
            labels[max_overlaps < fg_overlap_threshold] = 0
            # labels[np.logical_and(max_overlaps > bg_overlap_threshold,
            #                       max_overlaps < fg_overlap_threshold)] = -1

        match_labels.append(labels)
        match_inds.append(match)
        match_boxes.append(match_box)

    match_labels = np.concatenate(match_labels, axis=0)
    match_inds = np.concatenate(match_inds, axis=0)
    match_boxes = np.concatenate(match_boxes, axis=0)
    if boxes.is_cuda:
        return torch.from_numpy(match_labels).cuda(), \
               torch.from_numpy(match_inds).cuda(), \
               torch.from_numpy(match_boxes).cuda()
    return torch.from_numpy(match_labels), \
           torch.from_numpy(match_inds), \
           torch.from_numpy(match_boxes)


def decoding_box(deltas, anchors, box_encoding='fastrcnn'):

    boxes = anchors.view(-1, 4)
    deltas = deltas.view(-1, 4)
    n = deltas.size()[0]

    pred_boxes = deltas.clone()
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0

    if box_encoding == 'fastrcnn':

        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0] * 0.1
        dy = deltas[:, 1] * 0.1
        dw = deltas[:, 2] * 0.2
        dh = deltas[:, 3] * 0.2

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw + torch.log(widths))
        pred_h = torch.exp(dh + torch.log(heights))

        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h - 1

    elif box_encoding == 'simple':

        pred_boxes[:, 0] = deltas[:, 0] * widths + boxes[:, 0]
        pred_boxes[:, 1] = deltas[:, 1] * heights + boxes[:, 1]
        pred_boxes[:, 2] = deltas[:, 2] * widths + boxes[:, 2]
        pred_boxes[:, 3] = deltas[:, 3] * heights + boxes[:, 3]

    return pred_boxes


def encoding_box(gt_boxes, anchors, box_encoding='fastrcnn'):


    deltas = gt_boxes.clone()

    if box_encoding == 'fastrcnn':

        ex_widths = anchors[:, 2] - anchors[:, 0] + 1.0
        ex_heights = anchors[:, 3] - anchors[:, 1] + 1.0
        ex_ctr_x = anchors[:, 0] + 0.5 * ex_widths
        ex_ctr_y = anchors[:, 1] + 0.5 * ex_heights

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
        gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

        deltas[:, 0] = (gt_ctr_x - ex_ctr_x) / ex_widths / 0.1
        deltas[:, 1] = (gt_ctr_y - ex_ctr_y) / ex_heights / 0.1
        deltas[:, 2] = torch.log(gt_widths / ex_widths) / 0.2
        deltas[:, 3] = torch.log(gt_heights / ex_heights) / 0.2

    elif box_encoding == 'simple':

        ex_widths = anchors[:, 2] - anchors[:, 0] + 1.0
        ex_heights = anchors[:, 3] - anchors[:, 1] + 1.0

        deltas = gt_boxes - anchors
        deltas[:, 0] = deltas[:, 0] / ex_widths
        deltas[:, 1] = deltas[:, 1] / ex_heights
        deltas[:, 2] = deltas[:, 2] / ex_widths
        deltas[:, 3] = deltas[:, 3] / ex_heights

    return deltas


def apply_nms(boxes, scores, overlap_threshold):
    """
    boxes has to be a Variable
    for each image:
    apply boxes
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    dets = torch.cat((boxes, scores.view(-1, 1)), 1)
    order = scores.sort(0, descending=True)[1]
    n = boxes.size(0)
    keep = torch.LongTensor(n)
    num_out = torch.LongTensor(1)

    if not boxes.is_cuda:
      nms.cpu_nms(keep, num_out, dets.data, order, areas, overlap_threshold)
      return keep[:num_out[0]]

    else:
      dets = dets[order].contiguous()
      nms.gpu_nms(keep, num_out, dets.data, overlap_threshold)

    return order[keep[:num_out[0]].cuda()].contiguous()

def sample_rois(boxes, image_inds, gt_boxes_list,
                fg_overlap_threshold=0.5,
                rois_per_image=512,
                fg_fraction=0.25,
                ignore_threshold=0.2):
    """filter out ignored areas and keep the fg/bg ratio at 1:3"""
    boxes_np = boxes.data.cpu().numpy() if boxes.is_cuda else boxes.data.numpy()
    image_inds_np = image_inds.data.cpu().numpy() if image_inds.is_cuda else image_inds.data.numpy()

    num_boxes = boxes_np.shape[0]
    assert num_boxes == image_inds_np.size
    sampled_boxes = []
    sampled_probs = []
    sampled_labels = []
    sampled_image_inds = []
    batch_size = len(gt_boxes_list)

    for i, gt_boxes in enumerate(gt_boxes_list):
        boxes_im = boxes_np[image_inds_np == i]
        image_inds_im = image_inds_np[image_inds_np == i]

        keep_inds = filter_boxes(boxes_im)
        boxes_im = boxes_im[keep_inds]
        image_inds_im = image_inds_im[keep_inds]

        num_boxes_im = boxes_im.shape[0]
        labels = np.zeros((boxes_im.shape[0],), dtype=np.int64)

        # TODO: what if is no gt_boxes
        if gt_boxes.size > 0:
            # B x G
            overlaps = cython_bbox.bbox_overlaps(
                np.ascontiguousarray(boxes_im, dtype=np.float),
                np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

            gt_assignment = overlaps.argmax(axis=1)  # (B)
            max_overlaps = overlaps[np.arange(num_boxes_im), gt_assignment]

            labels[:] = gt_boxes[gt_assignment, 4]
            labels[max_overlaps < fg_overlap_threshold] = 0

            # ignoring areas
            ignored_mask = gt_boxes[:, 4] <= 0
            if np.any(ignored_mask):
                ignored_areas = gt_boxes[ignored_mask]
                ignored = cython_bbox.bbox_exclude_ignored_areas(
                    np.ascontiguousarray(boxes_im, dtype=np.float),
                    np.ascontiguousarray(ignored_areas[:, :4], dtype=np.float),
                    ignore_threshold
                )
                labels[ignored == 1] = -1

            # add ground-thruth boxes
            if True:
                valid_inds = np.where(gt_boxes[:, 4] > 0)[0]
                gb = gt_boxes[valid_inds][:, :4].astype(np.float32)
                gb = jitter_boxes(gb)
                cls = gt_boxes[valid_inds][:, 4].astype(np.int64)
                boxes_im = np.concatenate((boxes_im, gb), axis=0)
                labels = np.concatenate((labels, cls), axis=0)
                assert labels.shape[0] == boxes_im.shape[0]

                gn = gb.shape[0]

                new_inds = np.zeros((gn, ), dtype=image_inds_im.dtype) + i
                image_inds_im = np.concatenate((image_inds_im, new_inds), axis=0)
        else:
            labels = np.zeros((boxes_im.shape[0], ), dtype=np.float32)

        sampled_boxes.append(boxes_im[labels >= 0])
        sampled_labels.append(labels[labels >= 0])
        sampled_image_inds.append(image_inds_im[labels >= 0])

    sampled_boxes = np.concatenate(sampled_boxes, axis=0)
    sampled_labels = np.concatenate(sampled_labels, axis=0).astype(np.int64)
    sampled_image_inds = np.concatenate(sampled_image_inds, axis=0).astype(np.int64)

    # sampling
    bg_inds = np.where(sampled_labels == 0)[0]
    fg_inds = np.where(sampled_labels > 0)[0]
    # num_fg = min(fg_inds.size, 64)
    # if fg_inds.size > 0:
    #     fg_inds = np.random.choice(fg_inds, num_fg)
    if False:
        # sample all foregrounds
        num_fg = fg_inds.size
        num_bg = max(min(3 * num_fg, bg_inds.size), 16)
        if bg_inds.size > 0:
            bg_inds = np.random.choice(bg_inds, num_bg)
        keep_inds = np.append(fg_inds, bg_inds)
    else:
        # faster rcnn sampling
        num_fg = min(fg_inds.size, int(fg_fraction * rois_per_image * batch_size))
        if num_fg > 0:
            fg_inds = np.random.choice(fg_inds, num_fg, replace=False)
        num_bg = rois_per_image * batch_size - num_fg
        num_bg = min(num_bg, bg_inds.size)
        if bg_inds.size > 0:
            bg_inds = np.random.choice(bg_inds, num_bg, replace=False)
        keep_inds = np.append(fg_inds, bg_inds)

    sampled_labels = sampled_labels[keep_inds]
    sampled_boxes = sampled_boxes[keep_inds]
    sampled_image_inds = sampled_image_inds[keep_inds]

    # Guard against the case no sampled rois
    if sampled_labels.size == 0:
        sampled_boxes = boxes_np[:1, :]
        sampled_labels = np.array([-1], dtype=np.int64)
        sampled_image_inds = image_inds_np[:1].astype(np.int64)

    if boxes.is_cuda:
        return torch.from_numpy(sampled_boxes).cuda(), \
               torch.from_numpy(sampled_labels).cuda(), \
               torch.from_numpy(sampled_image_inds).cuda()
    return torch.from_numpy(sampled_boxes), \
           torch.from_numpy(sampled_labels), \
           torch.from_numpy(sampled_image_inds)


def jitter_boxes(boxes):
    num = boxes.shape[0]
    ws = boxes[:, 2] - boxes[:, 0] + 1.0
    hs = boxes[:, 3] - boxes[:, 1] + 1.0
    dws = (np.random.rand(num) - 0.5) * 0.2 * ws
    dhs = (np.random.rand(num) - 0.5) * 0.2 * hs
    boxes[:, 0] = boxes[:, 0] + dws
    boxes[:, 1] = boxes[:, 1] + dhs
    dws = (np.random.rand(num) - 0.5) * 0.2 * ws
    dhs = (np.random.rand(num) - 0.5) * 0.2 * hs
    boxes[:, 2] = boxes[:, 2] + dws
    boxes[:, 3] = boxes[:, 3] + dhs
    return boxes


def filter_boxes(boxes, min_size=8):
    ws = boxes[:, 2] - boxes[:, 0] + 1.0
    hs = boxes[:, 3] - boxes[:, 1] + 1.0
    inds = np.where(
        np.logical_and(ws >= min_size, hs >= min_size))[0]
    return inds
