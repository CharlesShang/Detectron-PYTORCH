#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import time

import torch.utils.data
import visdom

vis = visdom.Visdom()
# import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import libs.configs.config as cfg
from libs.nets.resnet import resnet50, resnet101
from libs.nets.model import detection_model
from libs.datasets.coco import get_loader
from libs.layers.data_layer import data_layer
from libs.utils.timer import Timer
from libs.nets.utils import everything2cuda, everything2numpy, \
    adjust_learning_rate, load_net, save_net

from libs.postprocessings.single_shot import compute_detection, draw_detection, draw_masks, \
    draw_gtboxes, draw_anchors, draw_images

_DEBUG = False

import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


args = parse_args()
cfg.load_from_yaml(args.cfg_file, cfg)

## data loader
train_data = get_loader(cfg.data_dir, cfg.split, data_layer,
                        is_training=True, batch_size=cfg.batch_size, num_workers=cfg.data_workers)
ANCHORS = np.vstack([anc.reshape([-1, 4]) for anc in train_data.dataset.ANCHORS])
class_names = train_data.dataset.classes
print('dataset len: {}'.format(len(train_data.dataset)))

pixels = np.zeros((cfg.num_classes, ), dtype=np.int64)
instances = np.zeros((cfg.num_classes, ), dtype=np.int64)

timer = Timer()
timer.tic()
for step, batch in enumerate(train_data):
    _, _, inst_masks, _, _, gt_boxes, _ = batch
    inst_masks = \
        everything2numpy(inst_masks)

    for j, gt_box in enumerate(gt_boxes):
        if gt_box.size > 0:
            cls = gt_box[:, -1].astype(np.int32)
            for i, c in enumerate(cls):
                instances[c] += 1
                m = inst_masks[j][i]
                pixels[c] += m.sum()
    t = timer.toc(False)

    if step % 500 == 0:
        print ('step: %d, instances: %d, pixels: %d, time: %.2fs' % (step, instances.sum(), pixels.sum(), t))

        with open("statistics", "wb") as f:
            pickle.dump({
                'pixels': pixels,
                'instances': instances,
                'class_names': class_names,
            }, f)
        pixel_dict = []
        for name, pixel in zip(class_names, pixels):
            pixel_dict.append([name, pixel])
        instance_dict = []
        for name, instance in zip(class_names, instances):
            instance_dict.append([name, instance])

        print('pixels', pixel_dict)
        print('instances', instance_dict)

with open("statistics", "wb") as f:
    pickle.dump({
        'pixels': pixels,
        'instances': instances,
        'class_names': class_names,
    }, f)
pixel_dict = []
for name, pixel in zip(class_names, pixels):
    pixel_dict.append([name, pixel])
instance_dict = []
for name, instance in zip(class_names, instances):
    instance_dict.append([name, instance])

print('pixels', pixel_dict)
print('instances', instance_dict)