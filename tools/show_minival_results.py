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
import ujson as json

res_file = 'output/diff_detection_result_minival2014_resnet50.json'
# res_file = 'output/res.json'

if __name__ == '__main__':

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

    with open(res_file, 'r') as f:
        all_results = json.load(f)

    test_data = get_loader(cfg.data_dir, cfg.split, data_layer,
                           is_training=False, batch_size=cfg.batch_size, num_workers=cfg.data_workers)
    coco = test_data.dataset

    coco.eval(res_file)

    # for step, batch in enumerate(test_data):
    #     input, image_ids, image_ori = batch
    #
    #
    #     for img_id, img in zip(image_ids, image_ori):
    #         # draw
    #         img_id