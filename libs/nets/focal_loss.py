#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class FocalLoss(_WeightedLoss):
    """
    implementation of paper "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    NOTE: This paper only introduce focal loss for binary classification.
        I adapt it to multiple classes.
        gamma controls the balance among well classified examples, and weight controls the inter class balance.
    """
    def __init__(self, weight=None, size_average=True, ignore_index=-100, gamma=2, alpha=0.25,
                 activation='sigmoid'):
        super(FocalLoss, self).__init__(weight)
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.alpha = alpha
        self.activation = activation
        self.size_average = size_average

    def softmax_loss(self, input, target):
        assert not target.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

        fg_nums = target.data.gt(0).sum().item()

        n, c = input.size()[0], input.size()[1]
        probs = F.softmax(input, dim=-1)
        dim0_inds = torch.arange(0, n).type(torch.cuda.LongTensor)
        dim1_inds = target.data.type(torch.cuda.LongTensor)
        probs = probs[dim0_inds, dim1_inds]
        # probs = probs.clamp(1e-10)
        alphas = target.data.gt(0).float() * self.alpha + target.data.eq(0).float() * (1 - self.alpha)
        fl = alphas * (probs - 1) ** self.gamma * F.cross_entropy(input, target, reduce=False)
        # fl = (probs - 1) ** self.gamma * F.cross_entropy(input, target, reduce=False)

        # if self.weight is not None:
        #   fl = fl * self.weight

        # if self.size_average:
        #     return fl.sum() / max(fg_nums, 10)
        # return torch.sum(fl)
        fl = fl.contiguous()
        if self.size_average:
            try:
                loss = fl.sum() / max(fg_nums, 10)
            except:
                print(n, c)
                print(alphas.size(), probs.size(), 'fg_nums:', fg_nums)
                print('probs nan:', torch.isnan(probs).any().item())
                print('target nan:', torch.isnan(target).any().item())
                print('fl nan:', torch.isnan(fl).any().item())
                print(target.size(), 'target labels: [%d, %d]' % (target.min().item(), target.max().item()))
                print(fl.max(), fl.min())
                print(dim1_inds.min(), dim1_inds.max())
                raise
            return loss
        else:
            return fl.sum()

    def sigmoid_loss(self, input, target):

        fg_nums = target.data.gt(0).sum().item()
        n, c = input.size(0), input.size(1)
        t = to_one_hot(target.data.cpu(), c + 1)  # [N,D]
        t = t[:, 1:]
        t = t.contiguous().cuda()

        p = input.sigmoid()
        pt = p.clone()
        pt[t == 0] = 1 - p[t == 0]  # pt = p if t>0 else 1-p
        # pt = pt.clamp(1e-10)

        alpha_t = self.alpha * target.gt(0).float() + (1 - self.alpha) * target.eq(0).float()
        loss = - (1 - pt).pow(self.gamma) * pt.log()
        loss = alpha_t * loss.sum(dim=1)

        try:
            ret = loss.sum() / max(fg_nums, 10) if self.size_average else loss.sum()
        except:
            print (pt.max(), pt.min())
            print (pt.log().min(), pt.log().max())
            raise
        return ret

    def forward(self, input, target):
        if self.activation == 'sigmoid':
            return self.sigmoid_loss(input, target)
        elif self.activation == 'softmax':
            return self.softmax_loss(input, target)
        else:
            raise ValueError('Unknown activation function: %s' % self.activation)


def to_one_hot(labels, num_classes):

    N = labels.size(0)
    y = torch.zeros(N, num_classes).long()
    y[torch.arange(N).long(), labels] = 1
    return y


class SigmoidCrossEntropy(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100,):
        super(SigmoidCrossEntropy, self).__init__(weight)
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, input, target):
        fg_nums = target.data.gt(0).sum()
        n, c = input.size(0), input.size(1)
        t = to_one_hot(target.data.cpu(), c + 1)  # [N,D]
        t = t[:, 1:]
        t = t.contiguous().cuda()
        # t = Variable(t).cuda()

        p = input.sigmoid()
        pt = p.clone()
        pt[t == 0] = 1 - p[t == 0]  # pt = p if t>0 else 1-p
        # pt = pt.clamp(1e-10)

        loss = - (1 - pt) * pt.log()
        loss = loss.sum(dim=1)
        ret = loss.mean() if self.size_average else loss.sum()
        return ret

