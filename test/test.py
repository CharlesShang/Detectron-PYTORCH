#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import time
import argparse
import torch.utils.data
import json
import copy

import tensorboardX as tbx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import libs.configs.config as cfg
from libs.nets.resnet import resnet50, resnet101
from libs.nets.maskrcnn import MaskRCNN
from libs.nets.retinatnet import RetinaNet

from libs.datasets.factory import get_data_loader
from libs.layers.data_layer import data_layer
from libs.utils.timer import Timer
from libs.nets.utils import everything2cuda, everything2numpy, load_net
import libs.postprocessings.single_shot as single_shot

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)


def log_images(imgs, names, global_step, prefix='Image'):
    summary = []
    for i, I in enumerate(imgs):
        summary.append(tbx.summary.image('%d/%s-%s'%(global_step, prefix, names[i]), I))
    return summary

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--restore', dest='restore',
                        help='optional config file', default='', type=str)
    parser.add_argument('--train_dir', dest='train_dir',
                        help='optional config file', default='', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    cfg.load_from_yaml(args.cfg_file, cfg)

    if args.restore and os.path.exists(args.restore):
        cfg.restore = args.restore
        print('using {} to restore'.format(args.restore))

    if args.train_dir:
        if not os.path.exists(args.train_dir): os.makedirs(args.train_dir)
        cfg.train_dir = args.train_dir
        print('using {} as train dir'.format(args.train_dir))

    return args


args = parse_args()

# data loader
cfg.batch_size = 1
split_test = cfg.split_test if cfg.split_test else cfg.split
get_loader = get_data_loader(cfg.datasetname)
test_data = get_loader(cfg.data_dir, split_test,
                       is_training=False, batch_size=cfg.batch_size,
                       num_workers=cfg.data_workers, shuffle=False)
dataset = test_data.dataset


def main():
    # config model and lr
    num_anchors = len(cfg.anchor_ratios) * len(cfg.anchor_scales[0]) * len(cfg.anchor_shift) \
        if isinstance(cfg.anchor_scales[0], list) else \
        len(cfg.anchor_ratios) * len(cfg.anchor_scales)

    resnet = resnet50 if cfg.backbone == 'resnet50' else resnet101
    detection_model = MaskRCNN if cfg.model_type.lower() == 'maskrcnn' else RetinaNet

    model = detection_model(resnet(pretrained=True, maxpool5=cfg.maxpool5),
                            num_classes=cfg.num_classes, num_anchors=num_anchors,
                            strides=cfg.strides, in_channels=cfg.in_channels, f_keys=cfg.f_keys,
                            num_channels=256, is_training=False, activation=cfg.class_activation)

    lr = cfg.lr
    start_epoch = 0
    if cfg.restore is not None:
        meta = load_net(cfg.restore, model)
        print(meta)
        if meta[0] >= 0:
            start_epoch = meta[0] + 1
            lr = meta[1]
        print('Restored from %s, starting from %d epoch, lr:%.6f' % (cfg.restore, start_epoch, lr))
    else:
        raise ValueError('restore is not set')

    model.cuda()
    model.eval()

    class_names = test_data.dataset.classes
    print('dataset len: {}'.format(len(test_data.dataset)))

    tb_dir = os.path.join(cfg.train_dir, cfg.backbone + '_' + cfg.datasetname, 'test', time.strftime("%h%d_%H"))
    writer = tbx.FileWriter(tb_dir)

    # main loop
    timer_all = Timer()
    timer_post = Timer()
    all_results1 = []
    all_results2 = []
    all_results_gt = []
    for step, batch in enumerate(test_data):

        timer_all.tic()

        # NOTE: Targets is in NHWC order!!
        # input, anchors_np, im_scale_list, image_ids, gt_boxes_list = batch
        # input = everything2cuda(input)
        input_t, anchors_np, im_scale_list, image_ids, gt_boxes_list = batch
        input = everything2cuda(input_t, volatile=True)

        outs = model(input, gt_boxes_list=None, anchors_np=anchors_np)

        if cfg.model_type == 'maskrcnn':
            rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts, anchors, \
            rois, roi_img_ids, rcnn_logit, rcnn_box, rcnn_prob, rcnn_labels, rcnn_bbtargets, rcnn_bbwghts = outs
            outputs = [rois, roi_img_ids, rpn_logit, rpn_box, rpn_prob, rcnn_logit, rcnn_box, rcnn_prob, anchors]
            targets = []
        elif cfg.model_type == 'retinanet':
            rpn_logit, rpn_box, rpn_prob, _, _, _ = outs
            outputs = [rpn_logit, rpn_box, rpn_prob]
        else:
            raise ValueError('Unknown model type: %s' % cfg.model_type)

        timer_post.tic()

        dets_dict = model.get_final_results(outputs, everything2cuda(anchors_np),
                                            score_threshold=0.01,
                                            max_dets=cfg.max_det_num * cfg.batch_size,
                                            overlap_threshold=cfg.overlap_threshold)
        if 'stage1' in dets_dict:
            Dets = dets_dict['stage1']
        else:
            raise ValueError('No stage1 results:', dets_dict.keys())
        Dets2 = dets_dict['stage2'] if 'stage2' in dets_dict else Dets

        t3 = timer_post.toc()
        t = timer_all.toc()

        formal_res1 = dataset.to_detection_format(copy.deepcopy(Dets), image_ids, im_scale_list)
        formal_res2 = dataset.to_detection_format(copy.deepcopy(Dets2), image_ids, im_scale_list)
        all_results1 += formal_res1
        all_results2 += formal_res2

        Dets_gt = []
        for gb in gt_boxes_list:
            cpy_mask = gb[:, 4] >= 1
            gb = gb[cpy_mask]
            n = cpy_mask.astype(np.int32).sum()
            res_gt = np.zeros((n, 6))
            res_gt[:, :4] = gb[:, :4]
            res_gt[:, 4] = 1.
            res_gt[:, 5] = gb[:, 4]
            Dets_gt.append(res_gt)
        formal_res_gt = dataset.to_detection_format(Dets_gt, image_ids, im_scale_list)
        all_results_gt += formal_res_gt

        if step % cfg.log_image == 0:
            input_np = everything2numpy(input)
            summary_out = []
            Is = single_shot.draw_detection(input_np, Dets, class_names=class_names)
            Is = Is.astype(np.uint8)
            summary_out += log_images(Is, image_ids, step, prefix='Detection/')

            Is = single_shot.draw_detection(input_np, Dets2, class_names=class_names)
            Is = Is.astype(np.uint8)
            summary_out += log_images(Is, image_ids, step, prefix='Detection2/')

            Imgs = single_shot.draw_gtboxes(input_np, gt_boxes_list, class_names=class_names)
            Imgs = Imgs.astype(np.uint8)
            summary_out += log_images(Imgs, image_ids, float(step), prefix='GT')

            for s in summary_out: writer.add_summary(s, float(step))

        if step % cfg.display == 0:
            print(time.strftime("%H:%M:%S ") +
                  'Epoch %d iter %d: speed %.3fs (%.3fs)' % (0, step, t, t3) +
                  ' ImageIds: ' + ', '.join(str(s) for s in image_ids), end='\r')

    res_dict = {
        'stage1': all_results1, 'stage2': all_results2, 'gt': all_results_gt
    }
    return res_dict


def eval(all_results, suffix=''):
    suffix = '_' + suffix if suffix != '' else suffix
    if cfg.datasetname in ['coco', 'detrac']:
        res_file = os.path.join('output', cfg.backbone + '_result_' +
                                cfg.datasetname + '_' + split_test + suffix + '.json')
        with open(res_file, 'w') as f:
            json.dump(all_results, f)
        dataset.eval(res_file)

    elif cfg.datasetname in ['pascal_voc', 'kitti_voc']:
        output_dir = os.path.join('output', cfg.datasetname)
        all_boxes = dataset.to_evaluation_format(all_results)
        dataset.evaluate_detections(all_boxes, )

    elif cfg.datasetname == 'citypersons':
        suffix = '_' + suffix if suffix != '' else suffix
        res_file = os.path.join(cfg.train_dir, 'retinanet_result_'
                                + cfg.datasetname + '_' + cfg.split_test
                                + '_' + cfg.backbone + suffix + '.json')
        with open(res_file, 'w') as f:
            json.dump(all_results, f, indent=1, separators=(',', ': '))
        dataset.eval(res_file)
        dataset.eval_over_scales(res_file)

    elif cfg.datasetname == 'caltech':
        suffix = '_' + suffix if suffix != '' else suffix
        res_file = os.path.join(cfg.train_dir, 'retinanet_result_'
                                + cfg.datasetname + '_' + cfg.split_test
                                + '_' + cfg.backbone + suffix + '.json')
        with open(res_file, 'w') as f:
            json.dump(all_results, f, indent=1, separators=(',', ': '))
        dataset.eval(res_file)
        dataset.eval_over_scales(res_file)

if __name__ == '__main__':

    res_dict = main()
    if 'stage1' in res_dict:
        print('')
        print('*** Result of Stage 1 ***')
        print('')
        eval(res_dict['stage1'], suffix='stage1')
    if 'stage2' in res_dict:
        print('')
        print('*** Result of Stage 2 ***')
        print('')
        eval(res_dict['stage2'], suffix='stage2')

    if 'gt' in res_dict:
        print('')
        print('*** Result of Stage GT ***')
        print('')
        eval(res_dict['gt'], suffix='gt')