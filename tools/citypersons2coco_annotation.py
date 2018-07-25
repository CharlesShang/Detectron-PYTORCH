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

# annot_path = './data/coco/annotations/instances_train2014.json'
# annot_path_new = './data/coco/annotations/instances_train2014_new.json'

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

def mat2coco_annots(annot_path):
    for split, annot_path in zip(['train', 'val'], ['data/citypersons/annotations/anno_train.mat',
                       'data/citypersons/annotations/anno_val.mat']):
        annots = sio.loadmat(annot_path)
        annots = annots['anno_' + split + '_aligned'].reshape([-1])

        coco = COCO()
        coco.dataset['images'] = []
        coco.dataset['categories'] = []
        coco.dataset['annotations'] = []
        coco.dataset['categories'].append({
            'name': 'pedestrian',
            'id': 1,
            'supercategory': 'person',
        })
        coco.dataset['categories'].append({
            'name': 'rider',
            'id': 2,
            'supercategory': 'person',
        })
        coco.dataset['categories'].append({
            'name': 'sitting_person',
            'id': 3,
            'supercategory': 'person',
        })
        coco.dataset['categories'].append({
            'name': 'other',
            'id': 4,
            'supercategory': 'person',
        })
        annid = 0
        for i, ann in enumerate(annots):
            ann = ann.reshape([-1])
            city_name, image_name, bbs = ann[0][0][0], ann[0][1][0], ann[0][2]
            coco.dataset['images'].append({
                'id': i,
                'file_name': image_name,
                'height': 1024,
                'width': 2048,
                'url': 'citypersons',
            })
            gt_classes, gt_boxes = bbs[:, 0], bbs[:, 1:5]
            areas = (gt_boxes[:, 2] + 1) * (gt_boxes[:, 3] + 1)
            heights = gt_boxes[:, 3].copy()
            gt_boxes[:, 2:4] += gt_boxes[:, 0:2]
            for j in range(gt_classes.size):
                bb = gt_boxes[j, :]
                x1, y1, x2, y2 = [bb[0], bb[1], bb[2], bb[3]]
                coco.dataset['annotations'].append({
                    'image_id': i,
                    'bbox': gt_boxes[j, :],
                    'category_id': gt_classes[j],
                    'id': annid + 1,
                    'iscrowd': 0,
                    'area': areas[j],
                    'segmentation': [[x1, y1, x1, y2, x2, y2, x2, y1]],
                    'height': heights[j],
                })
        coco.createIndex()
        return coco

if __name__ == '__main__':
    d = json.load(open(annot_path, 'r'))
    print(d.keys())
    print(d['annotations'][0].keys(), d['annotations'][0])
    print(d['categories'][0].keys(), d['categories'][0], len(d['categories']))
    print(d['images'][0], d['images'][0]['id'])

    annot_path_trainval = './data/coco/annotations/instances_trainval_minus2014.json'
    if not os.path.exists(annot_path_trainval):
        merge_trainval(annot_path_trainval)

    with open(annot_path_new, 'w') as f:
        json.dump(d, f, indent=1, separators=(',', ': '))


    train_annots = COCO(annot_path_trainval)
