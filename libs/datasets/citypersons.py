from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import scipy.io as sio
import numpy as np
import torch
import torchvision.datasets

import xml.etree.ElementTree as ET
import libs.configs.config as cfg
from .dataloader import sDataLoader
from .citypersons_eval import COCO, COCOeval
from libs.boxes.anchor import anchors_plane
from libs.nets.utils import everything2tensor
import libs.boxes.cython_bbox as cython_bbox
"""
dir structure:
./data/
        data/citypersons
        data/citypersons/cityscape
            data/citypersons/cityscape/leftImg8bit/{train|val|test}
            data/citypersons/cityscape/gtFine/{train|val|test}
"""
class citypersons(object):

    def __init__(self, data_root, split, data_handler=None, is_training=True):
        assert split in ['train', 'val', 'test'], \
            'unknow citypersons split settings: {}'.format(split)
        self.annot_path = os.path.join(data_root, 'annotations', 'anno_' + split + '.mat')
        assert os.path.exists(self.annot_path), \
            '{} not found'.format(self.annot_path)

        self.cityscape_path = os.path.join(data_root, 'cityscape')
        self.split = split
        self.gt_annots = []
        self.classes = ('__background__', # always index 0
                         'pedestrian', 'rider', 'sitting', 'unusual', 'group')

        self._data_handler = data_handler
        self._is_training = is_training
        self.ANCHORS = []
        self._build_anchors()
        self.load_gt()

    def _build_anchors(self):
        if len(self.ANCHORS) == 0:
            ih, iw = cfg.input_size
            all_anchors = []
            for i, stride in enumerate(cfg.strides):
                height, width = int(ih / stride), int(iw / stride)
                scales = cfg.anchor_scales[i] if isinstance(cfg.anchor_scales[i], list) else cfg.anchor_scales
                anchors = anchors_plane(height, width, stride,
                                            scales=scales,
                                            ratios=cfg.anchor_ratios,
                                            base=cfg.anchor_base)
                all_anchors.append(anchors)
            self.ANCHORS = all_anchors

    def load_gt(self):
        annots = sio.loadmat(self.annot_path)
        annots = annots['anno_'+self.split+'_aligned'].reshape([-1])
        for ann in annots:
            ann = ann.reshape([-1])
            city_name, image_name, bbs = ann[0][0][0], ann[0][1][0], ann[0][2]
            city_name = 'tubingen' if city_name == 'tuebingen' else city_name
            img_name = os.path.join(self.cityscape_path, 'leftImg8bit', self.split,
                                    city_name, image_name)
            if not os.path.exists(img_name):
                raise ValueError('image {} not found'.format(img_name))

            gt_classes, gt_boxes = bbs[:, 0], bbs[:, 1:5]
            gt_boxes_vis = bbs[:, 6:10]
            gt_boxes[:, 2:4] += gt_boxes[:, 0:2]
            gt_boxes_vis[:, 2:4] += gt_boxes_vis[:, 0:2]
            self.gt_annots.append({
                'img_name': img_name,
                'gt_boxes': gt_boxes,
                'gt_classes': gt_classes,
                'gt_boxes_vis': gt_boxes_vis,
            })

    def eval(self, resFile):
        """evaluation detection results"""
        annType = 'bbox'
        annFile = ''
        with open('results.txt' 'w') as res_file:
            for id_setup in range(0, 4):
                cocoGt = COCO(annFile)
                cocoDt = cocoGt.loadRes(resFile)
                imgIds = sorted(cocoGt.getImgIds())
                cocoEval = COCOeval(cocoGt, cocoDt, annType)
                cocoEval.params.imgIds = imgIds
                cocoEval.evaluate(id_setup)
                cocoEval.accumulate()
                cocoEval.summarize(id_setup, res_file)

    def __len__(self):
        return len(self.gt_annots)

    def __getitem__(self, i):
        annot = self.gt_annots[i]
        bboxes = annot['gt_boxes'].astype(np.float32)
        bboxes_vis = annot['gt_boxes_vis'].astype(np.float32)
        classes = annot['gt_classes'].astype(np.int32)
        img_name = annot['img_name']
        height, width = cv2.imread(img_name).shape[:2]
        # ignore regions and groups
        # for the original annotations: 0 denotes ignored regions, 5 groups
        classes[classes == 0] = -1
        classes[classes == 5] = -1

        # ignore object whose visible area is limited
        areas = ((bboxes_vis[:, 3] - bboxes_vis[:, 1] + 1) * (bboxes_vis[:, 2] - bboxes_vis[:, 0] + 1)).astype(np.float32) / \
                ((bboxes[:, 3] - bboxes[:, 1] + 1) * (bboxes[:, 2] - bboxes[:, 0] + 1)).astype(np.float32)
        classes[areas < 0.1] = -1


        n = classes.shape[0]
        inst_masks = np.zeros([n, height, width], dtype=np.int32)
        mask = np.zeros([height, width], dtype=np.int32)
        assert n == bboxes.shape[0] == inst_masks.shape[0]

        im, TARGETS, inst_masks, mask, ori_im, ANNOTATIONS = \
            self._data_handler(img_name, bboxes, classes, inst_masks, mask, self._is_training, self.ANCHORS)

        im = np.transpose(im, [2, 0, 1])  # c, h, w
        if self._is_training:
            ih, iw = im.shape[1], im.shape[2]
            # masks = np.transpose(masks, [0, 3, 1, 2]) # n,
            # mask = mask[np.newaxis, :, :] # 1xhxw
            downsampled_mask = []
            for _, stride in enumerate(cfg.strides):
                assert ih % stride == iw % stride == 0, '{} {} {}'.format(ih, iw, stride)
                h, w = ih // stride, iw // stride
                downsampled_mask.append(cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.int64))

            # to tensor
            im = everything2tensor(im.copy())
            TARGETS = everything2tensor(TARGETS)
            inst_masks = everything2tensor(inst_masks.astype(np.int64).copy())
            mask = everything2tensor(mask.astype(np.int64).copy())
            downsampled_mask = everything2tensor(downsampled_mask)
        else:
            """testing"""
            im = everything2tensor(im.copy())
            TARGETS = inst_masks = mask = downsampled_mask = ANNOTATIONS = []

        img_id = os.path.splitext(os.path.basename(img_name))[0]

        return im, TARGETS, inst_masks, mask, downsampled_mask, ori_im, ANNOTATIONS, img_id

def collate_fn(data):
    input = torch.stack([d[0] for d in data])
    mask = torch.stack([d[3] for d in data])
    inst_masks = [d[2] for d in data]  # a list of (N, H, W). N can be different

    gt_boxes = [np.hstack((d[6][0], d[6][1][:, np.newaxis])) for d in data]

    num_layers = len(data[0][4])
    downsampled_mask = []
    for i in range(num_layers):
        downsampled_mask.append(torch.stack([d[4][i] for d in data]))

    labels = torch.stack([d[1][0] for d in data])
    label_weights = torch.stack([d[1][1] for d in data])
    bbox_targets = torch.stack([d[1][2] for d in data])
    bbox_inside_weights = torch.stack([d[1][3] for d in data])
    Targets = [labels, label_weights, bbox_targets, bbox_inside_weights]

    # image ids
    image_ids = [d[7] for d in data]

    return input, Targets, inst_masks, mask, downsampled_mask, gt_boxes, image_ids

def collate_fn_testing(data):
    input = torch.stack([d[0] for d in data])

    # original images
    images_ori = [d[5] for d in data]

    # image ids
    image_ids = [d[7] for d in data]

    # gt_boxes = [np.hstack((d[6][0], d[6][1][:, np.newaxis])) for d in data]

    return input, image_ids, images_ori

def get_loader(data_dir, split, data_handler, is_training, batch_size = 16, shuffle=True, num_workers=4):

    dataset = citypersons(data_dir, split, data_handler=data_handler, is_training=is_training)
    if is_training:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)
    else:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn_testing)

if __name__ == '__main__':
    cfg.data_dir = './data/citypersons/'
    d = citypersons('./data/citypersons/', split='train')
    # res = d.roidb
    from IPython import embed; embed()