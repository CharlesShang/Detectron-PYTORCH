#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

"""RUN configs"""
cuda=True
display = 10
log_image = 200
restore = None # None or path to previous model
# restore = 'output/maskrcnn_resnet50_coco_ep3.h5'

"""TRAINING"""
start_over = False # if True, set start_ep to 0
max_epoch = 100
lr_decay_epoches = [10, 20, 25]
lr_decay = 0.1
lr = 0.001
momentum = 0.9
weight_decay = 0.0001
# rpn_cls, rpn_box, mask
loss_weights = [1.0, 1.0, 1.0]
# using focal loss
use_focal_loss = True
# use online hard example mining
use_ohem = True
mild_ohem = False
# loss weights of coordinates
bbweights = [1., 1., 1., 1.]
# SGD, RMS, Adam
solver = 'SGD'

# adding more weights to minority classes
# backgroud, person, bicycle, car, motorcycle, airplane, bus, train, truck, ...
class_weights = [0.5,  1. ,  2.6,  1.8,  2.5,  2.6,  2.6,  2.7,  2.4,  2.4,  2.4,  3.1,
        3.1,  3.3,  2.4,  2.5,  2.7,  2.6,  2.5,  2.4,  2.5,  2.6,  3.2,
        2.6,  2.6,  2.5,  2.4,  2.3,  2.6,  2.6,  3. ,  2.6,  3. ,  2.7,
        2.5,  2.9,  2.8,  2.6,  2.6,  2.7,  2. ,  2.5,  2.1,  2.7,  2.6,
        2.6,  2.2,  2.4,  2.6,  2.7,  2.6,  2.5,  2.5,  2.9,  2.6,  2.5,
        2.6,  1.8,  2.6,  2.5,  2.7,  2.2,  2.7,  2.6,  2.7,  3.1,  2.6,
        2.9,  2.6,  3.1,  2.8,  4. ,  2.6,  2.9,  2.1,  2.6,  2.6,  3.2,
        2.7,  4.1,  3.1]

# backgroud, person, bicycle, car, motorcycle, airplane, bus, train, truck, ...
segment_weights = [0.5,  1. ,  2.8,  2.2,  2.3,  2.4,  2.1,  2.1,  2.2,  2.6,  3.1,  2.9,
        2.9,  3.1,  2.4,  2.7,  2.1,  2.3,  2.4,  2.7,  2.5,  2.3,  2.7,
        2.4,  2.4,  3. ,  2.5,  2.9,  3.1,  2.5,  3.4,  3.5,  3.4,  3.8,
        3.1,  3.7,  3.6,  3.1,  2.9,  3.2,  2.7,  3. ,  2.5,  3.5,  3.3,
        3.5,  2.1,  2.6,  2.8,  2.4,  2.8,  2.6,  3. ,  2.8,  2. ,  2.7,
        2.3,  2.2,  2.2,  2.6,  2. ,  1.5,  2.5,  2.4,  2.4,  3.6,  3.2,
        2.8,  2.9,  3. ,  2.4,  4.1,  2.7,  2.4,  2.7,  2.9,  2.7,  3.4,
        2.4,  4.6,  3.7]


"""NETWORK"""
maxpool5=True
model_type='retinanet' # or maskrcnn
backbone= 'resnet50'
frozen=2 # [1,2,3,4,5] keep parameters fixed before this stage when training
is_training=True
num_classes=81
with_segment=True
# class activation, softmax or sigmoid?
# There's no background class for sigmoid
class_activation='sigmoid'
share_head=True
save_prefix=''
"""Data"""
# support coco, pascal_voc, citypersons
data_dir='data/coco/'
split='trainval2014'
split_test='minival2014_new'
data_workers=4
batch_size=6
input_size=(512, 512)
min_side=600
max_side=1000
canvas_width=1024
canvas_height=1024
keep_aspect_ratio=False
# using which layers
strides=(8, 16, 32, 64, 128)
f_keys=['C3', 'C4', 'C5', 'C6', 'C7']
in_channels=[512, 1024, 2048, 256, 256]

num_channels=256
use_augment = False
training_scale=[0.3, 0.5, 0.7, 1.0]

use_extend=False # only for citypersons

"""Anchors"""
base_size=256 # used to assign boxes to pyramid layers, corresponding to input size
rpn_bg_threshold=0.4
rpn_fg_threshold=0.6
rpn_batch_size=384
rpn_fg_fraction=0.25
# if this is set false, all objects are assigned with anchors even if no overlapping-conditions are satisfied
# update: 2017-oct-16, as SSD implementation, set default to true
rpn_clobber_positives=True
rpn_sample_strategy='simple' # 'simple', '', 'advanced'
rpn_box_encoding='fastrcnn' # 'linear', 'fastrcnn'

"""Anchor Output"""
ANCHORS=[]
anchor_scales=[2, 4, 8, 16, 32]
anchor_scales=[
    [2, 2.52, 3.17],
    [4, 5.04, 6.35],
    [8, 10.08, 12.70],
    [16, 20.16, 25.40],
    [32, 40.32, 50.80],]
anchor_ratios=[0.5, 1, 2.0] # height / width
anchor_base=16 # area is anchor_base x anchor_base
anchor_shift=[[0.0, 0.0],]  #shifting to [w*stride, h*stride]

"""Mask"""
mask_threshold=0.5
masks_per_image=64

"""ROIs and Mask"""
fg_threshold=0.5

"""ROIs"""
rois_per_image=128
roi_fg_fraction=0.25
roi_bg_threshold=0.3

"""Sample"""
min_size=4
pre_nms_top_n=12000
post_nms_top_n=3000

"""LOGs"""
train_dir='./output'
# support coco, pascal_voc, and citypersons
datasetname='coco'
tensorboard=True


"""Evaluation"""
max_det_num = 100
useCats = True
score_threshold = 0.5
# update: change overlap_threshold from 0.7 to 0.5, since 0.5 is better (0.2 mAP higher)...
overlap_threshold = 0.5
overlap_threshold_noncrowd = 0.4
# ignore an anchor if the covered area of an anchor is larger than this value
ignored_area_intersection_fraction = 0.3
noncrowd_classes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                    47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62,
                    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                    78, 79, 80]
score_threshold_classes = [-1,  0.7,  0.6,  0.65,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.5,
               0.4,  0.3,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.55,  0.45,  0.3, #22
               0.6,  0.6,  0.6 ,  0.5,  0.5,  0.45,  0.5,  0.3,  0.4,  0.6,  0.5, #33
               0.6,  0.5,  0.45,  0.5,  0.6,  0.5,  0.6,  0.6,  0.6,  0.6,  0.6,  #44
               0.4,  0.5,  0.6,  0.5,  0.45,  0.5,  0.6,  0.6,  0.4,  0.5,  0.6,  #55
               0.6,  0.6,  0.5,  0.6,  0.6,  0.6,  0.5,  0.5,  0.5,  0.4,  0.5,   #66
               0.4,  0.6,  0.4,  0.4,  0.3,  0.4,  0.3,  0.6,  0.5,  0.5,  0.3,   #77
               0.6,  0.3,  0.4] if datasetname == 'coco' else \
               [-1,
                0.5, 0.6, 0.6, 0.5,
                0.5, 0.5, 0.6, 0.5,
                0.5, 0.6, 0.4, 0.5,
                0.5, 0.5, 0.7, 0.5,
                0.6, 0.4, 0.4, 0.4]

import yaml

def load_from_yaml(yaml_file, cfg):
    with open(yaml_file) as f:
        cfg_yaml = yaml.load(f)
        print('')
        print('----------ymal----------')
        for k, v in cfg_yaml.iteritems():
            if hasattr(cfg, k):
                target = getattr(cfg, k)
                if (isinstance(target, tuple) or isinstance(target, list)) \
                      and isinstance(v, list):
                    setattr(cfg, k, v)
                elif isinstance(target, int) and isinstance(v, int):
                    setattr(cfg, k, v)
                elif isinstance(target, float) and isinstance(v, float):
                    setattr(cfg, k, v)
                elif isinstance(target, str) and isinstance(v, str):
                    setattr(cfg, k, v)
                elif isinstance(target, bool) and isinstance(v, bool):
                    setattr(cfg, k, v)
                elif target is None:
                    setattr(cfg, k, v)
                else:
                    raise ValueError('config file error {}-->{}'.format(v, target))
                print(k, ':', v)
        print('----------ymal----------')
        print('')
    return
