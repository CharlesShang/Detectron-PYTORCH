#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import libs.configs.config as cfg

from .head import RPNHead
from .model import detection_model
from .pyramid import PyramidFPN
from .rcnn import RCNN
from .focal_loss import FocalLoss, SigmoidCrossEntropy

from .smooth_l1_loss import smooth_l1_loss
from libs.layers.box import decoding_box, sample_rois
from libs.layers.anchor_target.modules.anchor_target import AnchorTarget
from libs.layers.roi_align_tf.pyramid_roi_align import PyramidRoIAlign, PyramidRoIAlign2
from libs.layers.roi_target.modules.roi_target import RoITarget
from libs.nets.utils import everything2cuda, everything2numpy
from libs.layers.data_layer import compute_rpn_targets_in_batch
import time

class MaskRCNN(detection_model):
    """treate RPN as a foreground/background binary classification task"""

    def __init__(self, backbone, num_classes, num_anchors,
                 strides=[4, 8, 16, 32],
                 in_channels=[256, 512, 1024, 2048],
                 f_keys=['C2', 'C3', 'C4', 'C5'],
                 num_channels=256,
                 is_training=True,
                 activation='sigmoid'):

        super(MaskRCNN, self).__init__(backbone, 2, num_anchors, is_training=is_training)

        assert len(strides) == len(in_channels) == len(f_keys)
        self.activation = activation
        self.num_classes = num_classes - 1 if self.activation == 'sigmoid' else num_classes

        self.maxpool5 = cfg.maxpool5
        self.pyramid = PyramidFPN(in_channels, f_keys, num_channels)

        # fg/bg classification
        self.rpn_activation = 'softmax'
        self.rpn_cls_loss_func = FocalLoss(gamma=2, alpha=0.25, activation=self.rpn_activation) if cfg.use_focal_loss \
            else nn.CrossEntropyLoss(ignore_index=-1)
        self.rpn = RPNHead(num_channels, 2, num_anchors, num_channels=256, activation=self.rpn_activation)

        self.pyramid_roi_align = PyramidRoIAlign2(7, 7)

        self.rcnn = RCNN(num_channels, num_classes, 7, 7, activation=self.activation)
        if self.activation == 'softmax':
            self.rcnn_cls_loss_func = nn.CrossEntropyLoss()
        elif self.activation == 'sigmoid':
            self.rcnn_cls_loss_func = SigmoidCrossEntropy()
        if is_training:
            self.roi_target = RoITarget(0.5)
            self.anchor_target = AnchorTarget(cfg.rpn_bg_threshold, cfg.rpn_fg_threshold, 0.2)

    def forward(self, input, gt_boxes_list, anchors_np, rpn_targets=None):

        batch_size = input.size(0)
        anchors = torch.from_numpy(anchors_np).cuda()
        endpoints = self.backbone(input)

        Ps = self.pyramid(endpoints)
        rpn_outs = []
        for i, f in enumerate(Ps):
            rpn_outs.append(self.rpn(f))

        rpn_logit, rpn_box = self._rerange(rpn_outs, last_dimension=2)
        rpn_prob = F.sigmoid(rpn_logit) if self.rpn_activation == 'sigmoid' else F.softmax(rpn_logit, dim=-1)
        rpn_prob.detach()

        if self.is_training:
            assert input.size(0) == len(gt_boxes_list), '%d vs %d' % (input.size(0), len(gt_boxes_list))
            if rpn_targets is None:
                rpn_targets = compute_rpn_targets_in_batch(gt_boxes_list, anchors_np)
                rpn_labels, _, rpn_bbtargets, rpn_bbwghts = everything2cuda(rpn_targets)
                # rpn_labels, rpn_bbtargets, rpn_bbwghts = self.compute_anchor_targets(anchors, gt_boxes_list)
            else:
                rpn_labels, rpn_bbtargets, rpn_bbwghts = rpn_targets

            rois, probs, roi_img_ids = self._stage_one_results(rpn_box, rpn_prob, anchors,
                                                               top_n=20000 * batch_size, overlap_threshold=0.7,
                                                               top_n_post_nms=2000)
            rois, roi_labels, roi_img_ids = sample_rois(rois, roi_img_ids, gt_boxes_list)
        else:
            rpn_labels = rpn_bbtargets = rpn_bbwghts = None
            rois, probs, roi_img_ids = self._stage_one_results(rpn_box, rpn_prob, anchors,
                                                               top_n=6000 * batch_size, overlap_threshold=0.7)
            rois, probs, roi_img_ids = self._thresholding(rois, probs, roi_img_ids, 0.05)

        rcnn_feats = self.pyramid_roi_align(Ps, rois, roi_img_ids)
        rcnn_logit, rcnn_box = self.rcnn(rcnn_feats)
        rcnn_prob = F.sigmoid(rcnn_logit) if self.activation == 'sigmoid' else F.softmax(rcnn_logit, dim=-1)
        rcnn_prob.detach()

        if self.is_training:
            rcnn_labels, rcnn_bbtargets, rcnn_bbwghts = self.compute_rcnn_targets(rois, roi_img_ids, gt_boxes_list)
            assert rcnn_labels.size(0) == rois.size(0) == roi_img_ids.size(0)
        else:
            rcnn_labels = rcnn_bbtargets = rcnn_bbwghts = None

        return rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts, anchors, \
               rois, roi_img_ids, rcnn_logit, rcnn_box, rcnn_prob, rcnn_labels, rcnn_bbtargets, rcnn_bbwghts

    def compute_rcnn_targets(self, rois, img_ids, gt_boxes_list):
        gt_img_inds = [np.zeros((gt.shape[0],), dtype=np.int64) + i for i, gt in enumerate(gt_boxes_list)]
        gt_img_inds = np.concatenate(gt_img_inds, axis=0)
        gt_boxes = np.concatenate(gt_boxes_list, axis=0).astype(np.float32)
        np.set_printoptions(precision=0, suppress=True)

        if gt_boxes.size > 0:
            gt_boxes = everything2cuda(gt_boxes).view(-1, 5)
            gt_img_inds = everything2cuda(gt_img_inds).view(-1)
            rcnn_labels, rcnn_bbtargets, rcnn_bbwgts = self.roi_target(rois, img_ids, gt_boxes, gt_img_inds)
        else:
            num_rois = rois.size(0)
            rcnn_labels = torch.LongTensor(num_rois).zero_().cuda()
            rcnn_bbtargets = torch.FloatTensor(num_rois, 4).zero_().cuda()
            rcnn_bbwgts = torch.FloatTensor(num_rois, 4).zero_().cuda()

        rcnn_bbwghts = rcnn_bbwgts.view(-1, 4)
        rcnn_bbox_targets = rcnn_bbtargets.view(-1, 4)
        return rcnn_labels, rcnn_bbox_targets, rcnn_bbwghts

    def compute_anchor_targets(self, anchors, gt_boxes_list):
        rpn_labels, rpn_bbtargets, rpn_bbwghts = self.anchor_target(anchors, gt_boxes_list)
        return rpn_labels, rpn_bbtargets, rpn_bbwghts

    def rcnn_cls_loss(self):
        return self.loss_dict['rcnn_cls_loss']

    def rcnn_box_loss(self):
        return self.loss_dict['rcnn_box_loss']

    def get_final_results(self, outputs, anchors,
                          score_threshold=0.1, max_dets=100, overlap_threshold=0.5):
        rois, roi_img_ids, rpn_logit, rpn_box, rpn_prob, rcnn_logit, rcnn_box, rcnn_prob = outputs[:8]
        Dets = self.get_final_results_stage1(rpn_box, rpn_prob, anchors,
                                             score_threshold=score_threshold,
                                             max_dets=max_dets,
                                             overlap_threshold=overlap_threshold)
        Dets2 = self.get_final_results_stage2(rcnn_box, rcnn_prob, roi_img_ids, rois,
                                              score_threshold=score_threshold,
                                              max_dets=max_dets,
                                              overlap_threshold=overlap_threshold)
        return {'stage1': Dets, 'stage2': Dets2}

    def get_final_results_stage2(self, rcnn_box, rcnn_prob, roi_img_ids, rois,
                                 score_threshold=0.1, max_dets=100, overlap_threshold=0.5):
        boxes, probs, img_ids, rois = \
            self._decoding_and_thresholding_stage2(rcnn_box, rcnn_prob, roi_img_ids, rois,
                                                   score_threshold=score_threshold)

        boxes, probs, img_ids, rois = \
            self._apply_nms_in_batch(boxes, probs, img_ids, rois,
                                     activation=self.activation,
                                     overlap_threshold=overlap_threshold)

        objness = self._objectness(probs, activation=self.activation)
        inds = objness.data.ge(score_threshold).nonzero().view(-1)

        if inds.numel() > max_dets:
            _, inds = objness.sort(dim=0, descending=True)
            inds = inds[:max_dets]
            boxes = boxes[inds]
            probs = probs[inds]
            img_ids = img_ids[inds]

        if torch.cuda.device_count() > 1:
            img_ids = img_ids.get_device() * (cfg.batch_size / torch.cuda.device_count()) + img_ids

        if self.activation == 'softmax':
            Dets = self.to_Dets(boxes, probs, img_ids)
        elif self.activation == 'sigmoid':
            Dets = self.to_Dets_sigmoid(boxes, probs, img_ids)
        else:
            raise ValueError('Unknown activation function %s' % self.rpn_activation)

        return Dets

    def build_losses_rcnn(self, rcnn_logit, rcnn_box, rcnn_labels, rcnn_bbtargets, rcnn_bbwghts):
        valid_inds = rcnn_labels.data.ge(0).nonzero().view(-1)
        rcnn_cls_loss = self.rcnn_cls_loss_func(rcnn_logit[valid_inds], rcnn_labels[valid_inds]) \
            if valid_inds.size(0) > 0 else torch.tensor(0.0, requires_grad=True).cuda()
        bb_nums = torch.sum(rcnn_bbwghts.data.gt(0).float())
        bbwght_outside = (rcnn_bbwghts > 0.0001).float() / max(bb_nums, 1.0)
        rcnn_box_loss = smooth_l1_loss(rcnn_box, rcnn_bbtargets, rcnn_bbwghts, bbwght_outside, sigma=1.0)
        return rcnn_cls_loss, rcnn_box_loss

    def loss(self, loss_weights=None):
        return self.loss_dict['rpn_cls_loss'] + self.loss_dict['rpn_box_loss'] + \
               self.loss_dict['rcnn_cls_loss'] + self.loss_dict['rcnn_box_loss']

    def build_losses(self, outputs, targets):
        rois, roi_img_ids, rpn_logit, rpn_box, rpn_prob, rcnn_logit, rcnn_box, rcnn_prob, anchors = outputs
        rpn_labels, rpn_bbtargets, rpn_bbwghts, rcnn_labels, rcnn_bbtargets, rcnn_bbwghts = targets
        rpn_labels[rpn_labels > 1] = 1
        if cfg.use_focal_loss:
            rpn_cls_loss, rpn_box_loss = \
                self.build_losses_rpn(rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts)
        else:
            rpn_cls_loss, rpn_box_loss = \
                self.build_losses_rpn_faster_rcnn(rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts)
            # rpn_cls_loss, rpn_box_loss = \
            #     self.build_losses_rpn(rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts)
        rcnn_cls_loss, rcnn_box_loss = \
            self.build_losses_rcnn(rcnn_logit, rcnn_box, rcnn_labels, rcnn_bbtargets, rcnn_bbwghts)
        self.loss_dict = {
            'rpn_cls_loss': rpn_cls_loss,
            'rpn_box_loss': rpn_box_loss,
            'rcnn_cls_loss': rcnn_cls_loss,
            'rcnn_box_loss': rcnn_box_loss,
        }
        return self.loss_dict

    def _decoding_and_thresholding_stage2(self, rcnn_box, rcnn_prob, roi_img_ids, rois,
                                          score_threshold=0.5):

        assert rois.size(0) == rcnn_box.size(0) == rcnn_prob.size(0) == roi_img_ids.size(0)

        rcnn_box = decoding_box(rcnn_box, rois, box_encoding=cfg.rpn_box_encoding)

        objness = self._objectness(rcnn_prob, activation=self.activation)
        inds = objness.data.ge(score_threshold).nonzero().view(-1)

        if inds.numel() == 0:
            _, inds = objness.sort(dim=0, descending=True)
            inds = inds[:1]

        rcnn_box = rcnn_box[inds]
        rcnn_prob = rcnn_prob[inds]
        roi_img_ids = roi_img_ids[inds]
        rois = rois[inds]

        return rcnn_box, rcnn_prob, roi_img_ids, rois
