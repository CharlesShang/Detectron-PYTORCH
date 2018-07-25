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
import libs.postprocessings.single_shot as single_shot

_DEBUG = False

import argparse
import pickle
import ujson as json

res_files = ['output/diff_detection_result_minival2014_resnet50.json',
            'output/diff_detection_result_minival2014_resnet50_10ep_512.json']
output_file = 'output/merged.json'

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

    test_data = get_loader(cfg.data_dir, cfg.split, data_layer,
                           is_training=False, batch_size=cfg.batch_size, num_workers=cfg.data_workers)
    coco = test_data.dataset

    args = parse_args()
    cfg.load_from_yaml(args.cfg_file, cfg)
    all_results = []
    for res_file in res_files:
        with open(res_file, 'r') as f:
            all_results += json.load(f)

    d = {u'bbox': [240.8, 170.4, 75.9, 64.3],
     u'category_id': 1,
     u'image_id': 532481,
     u'score': 0.68}
    img_map = {}
    for i, res in enumerate(all_results):
        if not img_map.has_key(res['image_id']):
            img_map[res['image_id']] = [res]
        else:
            img_map[res['image_id']] += [res]

    def xywh2xy(box):
        return [box[0], box[1], box[0]+box[2], box[1]+box[3]]

    def valid(box):
        return box[2] > 1 and box[3] > 1

    Dets = []
    image_ids = []
    for img_id in img_map.keys():
        res_list = img_map[img_id]
        dets = [
                xywh2xy(res['bbox']) + [res['score']] + [coco._cat_id_to_real_id(res['category_id'])]
                for res in res_list if valid(res['bbox'])
               ]
        dets = np.asarray(dets, dtype=np.float32)
        Dets.append(dets)
        image_ids.append(img_id)

    Dets = single_shot.nms_advanced(Dets)
    all_results = coco.to_detection_format(Dets, image_ids)

    with open(output_file, 'w') as f:
        json.dump(all_results, f)

    coco.eval(output_file)
