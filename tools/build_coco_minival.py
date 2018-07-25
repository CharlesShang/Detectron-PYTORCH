#!/usr/bin/env python
from __future__ import print_function

import os
import copy
import json as json
import numpy as np
import scipy.io as sio
from libs.datasets.pycocotools.coco import COCO

annot_path = './data/coco/annotations/instances_minival2014.json'
annot_path_new = './data/coco/annotations/instances_minival2014_new.json'

def merge_trainval(annot_path_trainval):
    annot_path_train = './data/coco/annotations/instances_train2014.json'
    annot_path_val   = './data/coco/annotations/instances_val2014.json'
    annot_path_minival = './data/coco/annotations/instances_minival2014.json'

    minival = json.load(open(annot_path_minival, 'r'))

    minival_set = set([img['id'] for img in minival['images']])
    train = json.load(open(annot_path_train, 'r'))
    val = json.load(open(annot_path_val, 'r'))

    images = []
    annotations = []
    cnt1, cnt2 = 0, 0
    for img in val['images']:
        if img['id'] not in minival_set:
            images.append(img)
        else:
            cnt1 += 1

    for ann in val['annotations']:
        if ann['image_id'] not in minival_set:
            annotations.append(ann)
        else:
            cnt2 += 1

    print(cnt1, cnt2)
    train['annotations'] += annotations
    train['images'] += images

    with open(annot_path_trainval, 'w') as f:
        json.dump(train, f, indent=1, separators=(',', ': '))
        print('saved')

def build_minival(annot_path_minval):
    annot_path_val   = './data/coco/annotations/instances_val2014.json'
    annot_path_minival = './data/coco/annotations/instances_minival2014.json'

    minival = json.load(open(annot_path_minival, 'r'))
    minival_new = minival

    minival_set = set([img['id'] for img in minival['images']])
    val = json.load(open(annot_path_val, 'r'))

    images = []
    annotations = []
    cnt1, cnt2 = 0, 0
    for img in val['images']:
        if img['id'] in minival_set:
            images.append(img)
        else:
            cnt1 += 1

    for ann in val['annotations']:
        if ann['image_id'] in minival_set:
            annotations.append(ann)
        else:
            cnt2 += 1

    minival_new['images'] = images
    minival_new['annotations'] = annotations


    with open(annot_path_minval, 'w') as f:
        json.dump(minival_new, f, indent=1, separators=(',', ': '))
        print('saved')


if __name__ == '__main__':
    d = json.load(open(annot_path, 'r'))
    print(d.keys())
    print(d['annotations'][0].keys(), d['annotations'][0])
    print(d['categories'][0].keys(), d['categories'][0], len(d['categories']))
    print(d['images'][0], d['images'][0]['id'])

    annot_path_trainval = './data/coco/annotations/instances_trainval_minus2014.json'
    if not os.path.exists(annot_path_trainval):
        merge_trainval(annot_path_trainval)

    annot_path_minival = './data/coco/annotations/instances_minival2014_new.json'
    if not os.path.exists(annot_path_minival):
        build_minival(annot_path_minival)


    # train_annots = COCO(annot_path_trainval)