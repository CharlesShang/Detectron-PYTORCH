#!/usr/bin/env python
# coding=utf-8
# This file is copied from torchvision.models
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX as tbx

import libs.configs.config as cfg

from .focal_loss import FocalLoss
from .smooth_l1_loss import smooth_l1_loss
from libs.layers.box import decoding_box, apply_nms
from libs.nets.utils import everything2numpy, everything2cuda


class detection_model(nn.Module):
    """
    This module apply backbone network, build a pyramid, then add rpns for all layers in the pyramid.
    """
    def __init__(self, backbone, num_classes, num_anchors, is_training=True, maxpool5=True):

        super(detection_model, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes  # number of classes for rpn
        self.num_anchors = num_anchors
        self.is_training = is_training
        self.rpn_activation = cfg.class_activation

        self.rpn_outs = []
        self.loss_dict = []

        self.with_segment = cfg.with_segment

        self._score_summaries = {}
        self._hist_summaries = {}
        self.global_step = 0
        self.anchors = None # must be set via running setup()

        self.maxpool5 = maxpool5

        if is_training:
            self.rpn_cls_loss_func = FocalLoss(gamma=2, alpha=0.25, activation=self.rpn_activation) \
                if cfg.use_focal_loss else nn.CrossEntropyLoss()

    def forward(self, input, gt_boxes_list, anchors_np):
        pass

    def _objectness(self, probs, activation=None):
        activation = self.rpn_activation if activation is None else activation
        if activation == 'softmax':
            return 1. - probs[:, 0]
        elif activation == 'sigmoid':
            return probs.max(dim=1)[0]
        else:
            raise ValueError('Unknown activation funtion %s' % self.activation)

    def _rerange(self, rpn_outs, last_dimension=None):
        """rerange (Pyramid, N, C, H, W) outputs to (NxLxHxW, C)"""
        last_dimension = self.num_classes if last_dimension is None else last_dimension
        n = rpn_outs[0][0].size()[0]
        c = rpn_outs[0][0].size()[1]
        cb = rpn_outs[0][1].size()[1]
        rpn_logit = [rpn[0].view(n, c, -1) for rpn in rpn_outs]
        rpn_box = [rpn[1].view(n, cb, -1) for rpn in rpn_outs]
        rpn_logit = torch.cat(rpn_logit, dim=2)
        rpn_box = torch.cat(rpn_box, dim=2)
        rpn_logit = rpn_logit.permute(0, 2, 1).contiguous().view(-1, last_dimension)
        num_endpoints = rpn_logit.size()[0]
        rpn_box = rpn_box.permute(0, 2, 1).contiguous().view(num_endpoints, -1)

        return rpn_logit, rpn_box

    def _stage_one_results(self, rpn_box, rpn_prob, anchors, top_n=2000,
                           overlap_threshold=0.7,
                           top_n_post_nms=None):
        boxes, probs, img_ids, anchors = \
            self._decode_and_choose_top_n_stage1(rpn_box, rpn_prob, anchors, top_n=top_n)
        boxes, probs, img_ids, anchors = \
            self._apply_nms_in_batch(boxes, probs, img_ids, anchors,
                                     activation=self.rpn_activation,
                                     overlap_threshold=overlap_threshold)
        if top_n_post_nms is not None:
            return boxes[:top_n_post_nms], probs[:top_n_post_nms], img_ids[:top_n_post_nms]
        return boxes, probs, img_ids

    def _thresholding(self, boxes, probs, batch_ids, score_threshold=0.1):
        objness = self._objectness(probs)
        inds = objness.data.ge(score_threshold).nonzero().view(-1)

        if inds.numel() == 0:
            _, inds = objness.sort(dim=0, descending=True)
            inds = inds[:10]

        boxes = boxes[inds]
        probs = probs[inds]
        batch_ids = batch_ids[inds]

        return boxes, probs, batch_ids

    def build_losses_rpn(self, rpn_logits, rpn_box, rpn_prob,
                         rpn_labels, rpn_bboxes, rpn_bbwghts):
        """With OHEM (Online Hard Example Mining)"""
        rpn_labels = rpn_labels.view(-1).long()

        assert rpn_logits.size()[0] == rpn_box.size()[0] == rpn_labels.size()[0], \
            'Dimension dont match %d vs %d vs %d' % (rpn_logits.size()[0], rpn_box.size()[0], rpn_labels.size()[0])

        if cfg.use_focal_loss:
            rpn_logits, rpn_labels = self._sample_valid(rpn_logits, rpn_labels)
        else:
            rpn_logits, rpn_labels = self._sample_OHEM(rpn_logits, rpn_labels, rpn_prob, rpn_box,
                                                       bg_fg_ratio=3)
        rpn_cls_loss = self.rpn_cls_loss_func(rpn_logits, rpn_labels)

        # build box loss
        rpn_bbwghts = rpn_bbwghts.view(-1, 4)
        rpn_bboxes = rpn_bboxes.view(-1, 4)
        bb_nums = torch.sum(rpn_bbwghts.data.gt(0).float())
        bbwght_outside = (rpn_bbwghts > 0.0001).float() / max(bb_nums, 1.0)
        rpn_box_loss = smooth_l1_loss(rpn_box, rpn_bboxes, rpn_bbwghts, bbwght_outside, sigma=1.0)

        return rpn_cls_loss, rpn_box_loss

    def build_losses_rpn_faster_rcnn(self, rpn_logits, rpn_box, rpn_prob,
                                     rpn_labels, rpn_bboxes, rpn_bbwghts):
        """No OHEM (Online Hard Example Mining)"""
        rpn_labels = rpn_labels.view(-1).long()

        assert rpn_logits.size()[0] == rpn_box.size()[0] == rpn_labels.size()[0], \
            'Dimension dont match %d vs %d vs %d' % (rpn_logits.size()[0], rpn_box.size()[0], rpn_labels.size()[0])

        rpn_logits, rpn_labels, all_rpn_labels = \
            self._sample_faster_rcnn(rpn_logits, rpn_labels, rpn_prob, rpn_box,
                                     rpn_batch_size=256, rpn_fg_fraction=0.5)

        rpn_cls_loss = F.cross_entropy(rpn_logits, rpn_labels, ignore_index=-1)

        # build box loss
        rpn_bbwghts = rpn_bbwghts.view(-1, 4)
        rpn_bboxes = rpn_bboxes.view(-1, 4)
        bb_nums = all_rpn_labels.eq(1).sum().item()
        bbwght_outside = all_rpn_labels.eq(1).float() / max(bb_nums * 4, 4.0)
        bbwght_outside = bbwght_outside.view(-1, 1)
        rpn_box_loss = smooth_l1_loss(rpn_box, rpn_bboxes, rpn_bbwghts, bbwght_outside, sigma=1.0)

        return rpn_cls_loss, rpn_box_loss

    def build_losses(self, outputs, targets):
        pass

    def loss(self):
        pass

    def cls_loss(self):
        return self.loss_dict['rpn_cls_loss']

    def box_loss(self):
        return self.loss_dict['rpn_box_loss']

    def _gather_fg(self, labels, boxes, logits):
        """choose all bgs, sort them, pick top_n bgs"""
        fg_inds = labels.data.ge(1).nonzero().view(-1)
        if fg_inds.numel() > 0:
            return labels[fg_inds], boxes[fg_inds], logits[fg_inds], fg_inds
        else:
            return None, None, None, fg_inds

    def _gather_bg(self, labels, probs, logits, top_n=2000):
        """choose all bgs, sort them, pick top_n bgs"""
        bg_inds = labels.data.eq(0).nonzero().view(-1)
        probs = probs[bg_inds]
        logits = logits[bg_inds]

        # objness = 1. - probs[:, 0]
        objness = self._objectness(probs)
        _, inds = objness.sort(dim=0, descending=True)
        top_n = min(top_n, inds.size(0))
        inds = inds[:top_n]
        return probs[inds], logits[inds], bg_inds[inds.data]

    def _sample_OHEM(self, rpn_logits, rpn_label, rpn_prob, rpn_boxes, bg_fg_ratio=3):

        rpn_prob.detach()

        fg_labels, fg_boxes, fg_logits, fg_inds = self._gather_fg(rpn_label, rpn_boxes, rpn_logits)
        fg_num = fg_inds.numel()
        top_n = max(fg_num * bg_fg_ratio, 16)
        bg_probs, bg_logits, bg_inds = self._gather_bg(rpn_label, rpn_prob, rpn_logits, top_n=top_n)
        bg_num = bg_inds.numel()

        # bg_objness = 1 - bg_probs[:, 0]
        bg_objness = self._objectness(bg_probs)

        if fg_inds is not None:
            chosen_inds = torch.cat((fg_inds, bg_inds), dim=0)
        else:
            chosen_inds = bg_inds

        labels = rpn_label[chosen_inds]

        if self.global_step % cfg.log_image == 0 and fg_num > 1:
            c = rpn_logits.size(1)
            sampled_fg_losses = 0.5 * torch.abs(self._to_one_hot(fg_labels, c) - rpn_prob[fg_inds]).sum(dim=1)
            self._score_summaries['Sample/PosLoss'] = sampled_fg_losses
            self._score_summaries['Sample/PosLossMax'] = sampled_fg_losses.max()

            bg_probs_all, _, _ = self._gather_bg(rpn_label, rpn_prob, rpn_logits, top_n=float('inf'))
            bg_objness_all = 1. - bg_probs_all[:, 0]
            self._score_summaries['Sample/NegLoss'] = bg_objness_all
            self._score_summaries['Sample/NegLoss_SampledMax'] = bg_objness.max()
            self._score_summaries['Sample/NegLoss_Sampled'] = bg_objness

            self._score_summaries['Sample/FG_nums'] = fg_num
            self._score_summaries['Sample/BG_nums'] = bg_num

        self.global_step += 1

        logits = rpn_logits[chosen_inds]
        return logits.contiguous(), labels.contiguous()

    def _sample_faster_rcnn_OHEM(self, rpn_logits, rpn_label, rpn_prob, rpn_boxes,
                      rpn_batch_size=256, rpn_fg_fraction=0.5):
        """Always sample rpn_batch_size examples. Even negative ones may dominate examples.
        Hopefully this is moderate than OHEM (FocalLoss > OHEM > this-sampler > _sample_faster_rcnn)
        """

        rpn_prob.detach()
        fg_inds = rpn_label.data.ge(1).nonzero().view(-1)
        fg_num = fg_inds.numel()
        fg_num_ = min(int(rpn_batch_size * rpn_fg_fraction), fg_num)
        if fg_num_ > 0:
            inds = torch.randperm(fg_num)[:fg_num_]
            fg_inds = fg_inds[inds]

        bg_inds = rpn_label.data.eq(0).nonzero().view(-1)
        bg_num = bg_inds.numel()
        bg_num_ = min(rpn_batch_size - fg_num_, bg_num)

        bg_probs, bg_logits, bg_inds = self._gather_bg(rpn_label, rpn_prob, rpn_logits, top_n=bg_num_)

        chosen_inds = torch.cat((fg_inds, bg_inds), dim=0)
        labels = rpn_label[chosen_inds]
        logits = rpn_logits[chosen_inds]

        all_labels = torch.zeros_like(rpn_label) - 1
        all_labels[fg_inds] = 1
        all_labels[bg_inds] = 0

        if self.global_step % cfg.log_image == 0 and fg_num > 1:
            self._score_summaries['Sample/FG_nums_total'] = fg_num
            self._score_summaries['Sample/BG_nums_total'] = bg_num
            self._score_summaries['Sample/FG_nums_train'] = fg_num_
            self._score_summaries['Sample/BG_nums_train'] = bg_num_
        self.global_step += 1

        return logits.contiguous(), labels.contiguous(), all_labels

    def _sample_faster_rcnn(self, rpn_logits, rpn_label, rpn_prob, rpn_boxes,
                            rpn_batch_size=256, rpn_fg_fraction=0.5):

        rpn_prob.detach()

        fg_inds = rpn_label.data.ge(1).nonzero().view(-1)
        fg_num = fg_inds.numel()
        fg_num_ = min(int(rpn_batch_size * rpn_fg_fraction), fg_num)
        if fg_num_ > 0:
            inds = torch.randperm(fg_num)[:fg_num_]
            fg_inds = fg_inds[inds]

        bg_inds = rpn_label.data.eq(0).nonzero().view(-1)
        bg_num = bg_inds.numel()
        bg_num_ = min(rpn_batch_size - fg_num_, bg_num)
        if bg_num_ > 0:
            inds = torch.randperm(bg_num)[:bg_num_]
            bg_inds = bg_inds[inds]

        chosen_inds = torch.cat((fg_inds, bg_inds), dim=0)
        labels = rpn_label[chosen_inds]
        logits = rpn_logits[chosen_inds]

        all_labels = torch.zeros_like(rpn_label) - 1
        all_labels[fg_inds] = 1
        all_labels[bg_inds] = 0

        if self.global_step % cfg.log_image == 0 and fg_num > 1:
            self._score_summaries['Sample/FG_nums_total'] = fg_num
            self._score_summaries['Sample/BG_nums_total'] = bg_num
            self._score_summaries['Sample/FG_nums_train'] = fg_num_
            self._score_summaries['Sample/BG_nums_train'] = bg_num_
        self.global_step += 1

        return logits.contiguous(), labels.contiguous(), all_labels

    def _sample_valid(self, rpn_logits, rpn_labels):

        # rpn_prob.detach()
        valid_inds = rpn_labels.data.ge(0).nonzero().view(-1)
        logits, labels = rpn_logits[valid_inds], rpn_labels[valid_inds]

        return logits.contiguous(), labels.contiguous()

    def _decode_and_choose_top_n_stage1(self, rpn_box, rpn_prob, anchors, top_n=1000):

        objness = self._objectness(rpn_prob)
        _, inds = objness.sort(dim=0, descending=True)
        inds = inds[:top_n]

        selected_boxes = rpn_box[inds]
        selected_probs = rpn_prob[inds]
        anchor_ids = inds % anchors.size(0)
        selected_anchors = anchors[anchor_ids]
        selected_boxes = decoding_box(selected_boxes, selected_anchors, box_encoding=cfg.rpn_box_encoding)
        selected_img_ids = inds / anchors.size(0)

        return selected_boxes, selected_probs, selected_img_ids, selected_anchors

    def _decoding_and_thresholding_stage1(self, rpn_box, rpn_prob, anchors, score_threshold=0.3, max_dets=100):

        selected_boxes, selected_probs, selected_img_ids, selected_anchors = \
            self._decode_and_choose_top_n_stage1(rpn_box, rpn_prob, anchors, top_n=max_dets * 3)

        objness = self._objectness(selected_probs)
        inds = objness.data.ge(score_threshold).nonzero().view(-1)

        if inds.numel() == 0:
            _, inds = objness.sort(dim=0, descending=True)
            inds = inds[:1]

        selected_boxes = selected_boxes[inds]
        selected_probs = selected_probs[inds]
        selected_img_ids = selected_img_ids[inds]
        selected_anchors = selected_anchors[inds]

        return selected_boxes, selected_probs, selected_img_ids, selected_anchors

    @staticmethod
    def _apply_nms_in_batch(boxes, probs, img_ids, anchors, activation, overlap_threshold=0.5):
        """apply non-maximum suppression for multiple images in a mini-batch"""
        objness = probs.max(dim=1)[0] if activation == 'sigmoid' else 1. - probs[:, 0]
        nmax = img_ids.max().cpu().data.numpy()
        nmin = img_ids.min().cpu().data.numpy()
        all_keeps = []
        for i in range(nmin, nmax + 1):
            inds = img_ids.data.eq(i).nonzero().view(-1)
            if inds.numel() > 0:
                keeps = apply_nms(boxes[inds][:, :4], objness[inds], overlap_threshold=overlap_threshold)
                all_keeps.append(inds[keeps])
        all_keeps = torch.cat(all_keeps, dim=0) if len(all_keeps) > 1 else all_keeps[0]
        return boxes[all_keeps], probs[all_keeps], img_ids[all_keeps], anchors[all_keeps]

    @staticmethod
    def to_Dets(boxes, probs, img_ids):
        """for each bbox, assign the class with the max prob"""
        boxes, probs, img_ids = everything2numpy([boxes, probs, img_ids])
        Dets = []
        for i in range(0, cfg.batch_size):
            inds = np.where(img_ids == i)[0]
            probs_ = probs[inds]
            boxes_ = boxes[inds]
            if probs_.shape[1] == 2:
                cls_ids = np.ones((probs_.shape[0], ), dtype=np.int32)
                cls_probs = probs_[:, 1]
            else:
                cls_ids = probs_[:, 1:].argmax(axis=1) + 1
                cls_probs = probs_[np.arange(probs_.shape[0]), cls_ids]

            dets = np.concatenate((boxes_.reshape(-1, 4),
                                   cls_probs[:, np.newaxis],
                                   cls_ids[:, np.newaxis]), axis=1)

            Dets.append(dets)
        return Dets

    @staticmethod
    def to_Dets_sigmoid(boxes, probs, img_ids):
        """for each bbox, assign the class with the max prob,
        NOTE: there is no background class, so the implementation is slightly different"""
        boxes, probs, img_ids = everything2numpy([boxes, probs, img_ids])
        Dets = []
        for i in range(0, cfg.batch_size):
            inds = np.where(img_ids == i)[0]
            probs_ = probs[inds]
            boxes_ = boxes[inds]
            if probs_.ndim == 1 or probs_.shape[1] == 1:
                cls_ids = np.ones((probs_.shape[0], ), dtype=np.int32)
                cls_probs = probs_.view(-1)
            else:
                cls_ids = probs_.argmax(axis=1) + 1
                cls_probs = probs_.max(axis=1)

            dets = np.concatenate((boxes_.reshape(-1, 4),
                                   cls_probs[:, np.newaxis],
                                   cls_ids[:, np.newaxis]), axis=1)

            Dets.append(dets)
        return Dets

    @staticmethod
    def to_Dets2(boxes, probs, img_ids, score_threshold=0.1):
        """for each box, there may be more than one class labels"""
        boxes, probs, img_ids = everything2numpy([boxes, probs, img_ids])
        Dets = []
        for i in range(0, cfg.batch_size):
            inds = np.where(img_ids == i)[0]
            probs_ = probs[inds]
            boxes_ = boxes[inds]
            if probs_.shape[1] == 2:
                cls_ids = np.ones((probs_.shape[0], ), dtype=np.int32)
                cls_probs = probs_[:, 1]
                dets = np.concatenate((boxes_.reshape(-1, 4),
                                       cls_probs[:, np.newaxis],
                                       cls_ids[:, np.newaxis]), axis=1)
            else:
                d0_inds, d1_inds = np.where(probs_[:, 1:] > score_threshold)
                if d0_inds.size > 0:
                    cls_ids = d1_inds + 1
                    cls_probs = probs_[d0_inds, cls_ids]
                    boxes_ = boxes_[d0_inds, :]
                    dets = np.concatenate((boxes_.reshape(-1, 4),
                                           cls_probs[:, np.newaxis],
                                           cls_ids[:, np.newaxis]), axis=1)
                else:
                    cls_ids = probs_[:, 1:].argmax(axis=1) + 1
                    cls_probs = probs_[np.arange(probs_.shape[0]), cls_ids]
                    dets = np.concatenate((boxes_.reshape(-1, 4),
                                           cls_probs[:, np.newaxis],
                                           cls_ids[:, np.newaxis]), axis=1)

            Dets.append(dets)
        return Dets

    @staticmethod
    def to_Dets2_sigmoid(boxes, probs, img_ids, score_threshold=0.1):
        boxes, probs, img_ids = everything2numpy([boxes, probs, img_ids])
        Dets = []
        for i in range(0, cfg.batch_size):
            inds = np.where(img_ids == i)[0]
            probs_ = probs[inds]
            boxes_ = boxes[inds]
            if probs_.ndim == 1 or probs_.shape[1] == 1:
                cls_ids = np.ones((probs_.shape[0], ), dtype=np.int32)
                cls_probs = probs_.view(-1)
                dets = np.concatenate((boxes_.reshape(-1, 4),
                                       cls_probs[:, np.newaxis],
                                       cls_ids[:, np.newaxis]), axis=1)
            else:
                d0_inds, d1_inds = np.where(probs_ > score_threshold)
                if d0_inds.size > 0:
                    cls_ids = d1_inds + 1
                    cls_probs = probs_[d0_inds, d1_inds]
                    boxes_ = boxes_[d0_inds, :]
                    dets = np.concatenate((boxes_.reshape(-1, 4),
                                           cls_probs[:, np.newaxis],
                                           cls_ids[:, np.newaxis]), axis=1)
                else:
                    cls_ids = probs_.argmax(axis=1) + 1
                    cls_probs = probs_[np.arange(probs_.shape[0]), cls_ids - 1]
                    dets = np.concatenate((boxes_.reshape(-1, 4),
                                           cls_probs[:, np.newaxis],
                                           cls_ids[:, np.newaxis]), axis=1)

            Dets.append(dets)
        return Dets

    def get_final_results(self, outputs, anchors, **kwargs):
        pass

    def get_final_results_stage1(self, rpn_box, rpn_prob, anchors,
                                 score_threshold=0.1,
                                 max_dets=100,
                                 overlap_threshold=0.5):

        selected_boxes, selected_probs, selected_img_ids, selected_anchors = \
            self._decoding_and_thresholding_stage1(rpn_box, rpn_prob, anchors,
                                                   score_threshold=score_threshold,
                                                   max_dets=max_dets * 3)

        selected_boxes, selected_probs, selected_img_ids, selected_anchors = \
            self._apply_nms_in_batch(selected_boxes, selected_probs,
                                     selected_img_ids, selected_anchors,
                                     activation=self.rpn_activation,
                                     overlap_threshold=overlap_threshold)

        if self.rpn_activation == 'softmax':
            Dets = self.to_Dets2(selected_boxes, selected_probs, selected_img_ids, score_threshold)
        elif self.rpn_activation == 'sigmoid':
            Dets = self.to_Dets2_sigmoid(selected_boxes, selected_probs, selected_img_ids, score_threshold)
        else:
            raise ValueError('Unknown activation function %s' % self.rpn_activation)

        return Dets

    def get_pos_anchors(self, score_threshold=0.1, max_dets=100):
        _, selected_probs, selected_img_ids, selected_anchors = \
            self._decoding_and_thresholding_stage1(score_threshold=score_threshold, max_dets=max_dets)

        if self.rpn_activation == 'softmax':
            Dets = self.to_Dets(selected_anchors, selected_probs, selected_img_ids)
        elif self.rpn_activation == 'sigmoid':
            Dets = self.to_Dets_sigmoid(selected_anchors, selected_probs, selected_img_ids)
        else:
            raise ValueError('Unknown activation function %s' % self.rpn_activation)

        return Dets

    def _to_one_hot(self, y, num_classes):
        c = num_classes + 1 if self.rpn_activation == 'sigmoid' else num_classes
        y_ = torch.FloatTensor(y.size()[0], c).zero_()
        y_ = y_.scatter_(1, y.view(-1, 1).data.cpu(), 1.0).cuda()
        if self.rpn_activation == 'sigmoid':
            y_ = y_[:, 1:]
        if y.is_cuda:
            y_ = y_.cuda()
        return y_

    def de_frozen_backbone(self):
        self.backbone.de_frozen()

    def _add_scalar_summary(self, key, tensor):
        if isinstance(tensor, torch.Tensor):
            return tbx.summary.scalar(key + '/L1', torch.abs(tensor).mean().data.cpu().numpy())
        elif isinstance(tensor, float) or isinstance(tensor, int):
            return tbx.summary.scalar(key, tensor)

    def _add_hist_summary(self, key, tensor):
        return tbx.summary.histogram(key, tensor.data.cpu().numpy(), bins='auto')

    def get_summaries(self, is_training=True):
        """
        Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
        """
        summaries = []

        for key, var in self._score_summaries.items():
            summaries.append(self._add_scalar_summary(key, var))
        self._score_summaries = {}
        # Add act summaries
        # for key, var in self._hist_summaries.items():
        #     summaries += self._add_hist_summary(key, var)
        self._hist_summaries = {}
        # Add train summaries
        if is_training:
            for k, var in dict(self.named_parameters()).items():
                if var.requires_grad:
                    # summaries.append(self._add_hist_summary(k, var))
                    summaries.append(self._add_scalar_summary('Params/' + k, var))
                    summaries.append(self._add_scalar_summary('Grads/' + k, var.grad))
        return summaries
