import os
import glob
import cv2
import numpy as np

import torch
import torch.utils.data as data

from dataloader import sDataLoader
import libs.configs.config as cfg
from libs.nets.utils import everything2tensor
from libs.boxes.anchor import anchors_plane


class ImageFolder(data.Dataset):

    classes = ('')

    def __init__(self, data_dir, split, data_handler, is_training=False):
        self._data_dir = data_dir
        self._split = split
        self._data_handler = data_handler
        self._image_list = []
        self._is_training = False
        self.ANCHORS = []
        self._load()
        self._build_anchors()
        self.classes = [str(i) for i in range(cfg.num_classes + 1)]

    def _load(self):
        self._image_list = glob.glob(os.path.join(self._data_dir, '*.jpg'))
        self._image_list.extend(glob.glob(os.path.join(self._data_dir, '*.png')))
        self._image_list = sorted(self._image_list)

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

    def to_detection_format(self, Dets, image_ids, ori_sizes = None):
        """Add a detection results to list"""
        list = []
        for i, (dets, img_id) in enumerate(zip(Dets, image_ids)):
            for box in dets:
                if ori_sizes is not  None:
                    size = ori_sizes[i]
                    box[0:4:2] = box[0:4:2] * size[1] / cfg.input_size[1]
                    box[1:4:2] = box[1:4:2] * size[0] / cfg.input_size[0]
                x, y = box[0], box[1]
                width, height = box[2] - box[0], box[3] - box[1]
                score, id = box[4], int(box[5])
                dict = {
                    "image_id": img_id,
                    "category_id": id,
                    "bbox": [round(x, 1), round(y, 1), round(width, 1), round(height, 1)],
                    "score": round(score, 3)
                }
                list.append(dict)

        return list

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, i):
        img_name = self._image_list[i]
        im = cv2.imread(img_name)
        height, width, _ = im.shape

        # empty annotations
        bboxes = np.asarray([[0, 0, width - 1, height - 1]], dtype=np.float32)
        classes = np.asarray((-1, ), dtype=np.int32)
        inst_masks = np.zeros([1, height, width], dtype=np.int32)
        mask = np.zeros([height, width], dtype=np.int32)

        im, TARGETS, inst_masks, mask, ori_im, ANNOTATIONS = \
            self._data_handler(img_name, bboxes, classes, inst_masks, mask, self._is_training, self.ANCHORS)

        im = np.transpose(im, [2, 0, 1])  # c, h, w
        im = everything2tensor(im.copy())
        TARGETS = inst_masks = mask = downsampled_mask = []
        img_id = os.path.split(img_name)[1]

        return im, TARGETS, inst_masks, mask, downsampled_mask, ori_im, ANNOTATIONS, img_id


def collate_fn_testing(data):
    input = torch.stack([d[0] for d in data])

    # original images
    images_ori = [d[5] for d in data]

    # image ids
    image_ids = [d[7] for d in data]

    gt_boxes_list = [np.hstack((d[6][0], d[6][1][:, np.newaxis])) for d in data]

    return input, image_ids, gt_boxes_list, images_ori


def get_loader(data_dir, split, data_handler, is_training, batch_size=1, shuffle=False, num_workers=2):
    assert is_training is False
    dataset = ImageFolder(data_dir, split, data_handler, is_training)
    return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn_testing)
