#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import libs.configs.config as cfg
from .roi_align import RoIAlignAvg as RoIAlign


class PyramidRoIAlign(nn.Module):
    """crop feature in a pyramid
    1. find the right layer
    2. apply roi align cropping
    """
    def __init__(self, aligned_height, aligned_width):
        super(PyramidRoIAlign, self).__init__()

        spatial_scales = np.asarray([1.0 / i for i in cfg.strides], dtype=np.float32)
        ref_areas = (np.asarray(cfg.anchor_scales, dtype=np.float32).mean(axis=1) * cfg.anchor_base) ** 2
        assert ref_areas.size == spatial_scales.size
        self.num_levels = ref_areas.shape[0]

        self.spatial_scales = spatial_scales
        self.ref_areas = torch.from_numpy(ref_areas).cuda()

        self.aligned_height = aligned_height
        self.aligned_width = aligned_width
        self.roi_align = RoIAlign(aligned_height, aligned_width)

    def forward(self, pyramids, bboxes, batch_inds):
        num_rois = bboxes.size()[0]
        order = torch.arange(0, num_rois).long().cuda()
        areas = (bboxes[:, 2] - bboxes[:, 0] + 1.0) * (bboxes[:, 3] - bboxes[:, 1] + 1.0)
        ratios = areas.view(-1, 1).expand(num_rois, self.num_levels) / self.ref_areas
        log_ratios = torch.abs(torch.log(torch.pow(ratios, 0.5)) / 0.6931472)
        log_ratios.detach()
        _, levels = log_ratios.min(dim=1)

        rois = torch.cat((batch_inds.view(-1, 1).float(), bboxes), dim=1)
        rois.detach()

        res = []
        ori_order = []
        rois_new = []
        for level in range(self.num_levels):
            assigned = levels.data.eq(level).nonzero().view(-1)
            if assigned.numel() > 0:
                rois_ = rois[assigned]
                f = self.roi_align(pyramids[level], rois_, self.spatial_scales[level])
                res.append(f)
                ori_order.append(order[assigned])
                rois_new.append(rois_)

        aligned_features = torch.cat(res, dim=0)
        ori_order = torch.cat(ori_order, dim=0)
        aligned_features[ori_order] = aligned_features.clone()
        rois_new = torch.cat(rois_new, dim=0)
        rois_new[ori_order] = rois_new.clone()

        return aligned_features


if __name__ == '__main__':

    # build pyramid and bboxes, and batch_inds
    bboxes = [
        [100, 100, 120, 130],
        [10, 10, 70, 70],
        [200, 200, 350, 350],
        [0, 0, 250, 250],
        [0, 0, 500, 500],
    ]
    bboxes = np.asarray(bboxes, dtype=np.float32)

    perm = np.random.permutation(bboxes.shape[0])

    pyramid = []
    batch_inds = []
    for i, s in enumerate(cfg.strides):
        t = torch.zeros((2, 1, 512 // s, 512 // s))
        ind = np.random.randint(0, 2)
        box = bboxes[i] // s
        print (s, ind, box, t.size())
        t[ind, :, int(box[1]):int(box[3]+1), int(box[0]):int(box[2]+1)] = i
        f = t.cuda()
        # f.requires_grad = True
        pyramid.append(f)
        batch_inds.append(ind)

    batch_inds = np.asarray(batch_inds, dtype=np.float32)
    pyramid_crop = PyramidRoIAlign(5, 5)
    aligned_features = pyramid_crop(pyramid, torch.from_numpy(bboxes).cuda(),
                                    torch.from_numpy(batch_inds).cuda())
    aligned_features.cpu()
    print(aligned_features.size())
    print(aligned_features)

    bboxes_ = bboxes[perm]
    batch_inds_ = batch_inds[perm]
    aligned_features = pyramid_crop(pyramid, torch.from_numpy(bboxes_).cuda(),
                                    torch.from_numpy(batch_inds_).cuda())

    # aligned_features.requires_grad = True
    f = aligned_features.cpu().data.numpy()
    print(aligned_features.size())
    print(aligned_features)
    print(aligned_features.requires_grad)
    print(bboxes_)



