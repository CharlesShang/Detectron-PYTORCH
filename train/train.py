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
import torch.nn as nn
import tensorboardX as tbx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import libs.configs.config as cfg
import libs.postprocessings.single_shot as single_shot

from libs.nets.resnet import resnet50, resnet101
from libs.nets.maskrcnn import MaskRCNN
from libs.nets.retinatnet import RetinaNet
from libs.nets.data_parallel import ListDataParallel, ScatterList

from libs.datasets.factory import get_data_loader
from libs.utils.timer import Timer
from libs.nets.utils import everything2cuda, everything2numpy, \
    adjust_learning_rate, load_net, save_net

torch.backends.cudnn.benchmark = True
seed = 42
torch.manual_seed(42)
np.random.seed(42)


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
    parser.add_argument('--solver', dest='solver',
                        help='SGD, RMS, Adam', default='', type=str)
    parser.add_argument('--focal_loss', dest='focal_loss',
                        help='using focal loss or not', action='store_true')
    parser.add_argument('--no-focal_loss', dest='focal_loss',
                        help='using focal loss or not', action='store_false')
    parser.set_defaults(focal_loss=True)
    parser.add_argument('--lr', dest='lr',
                        help='learning rate', default=0, type=float)
    parser.add_argument('--save_prefix', dest='save_prefix',
                        help='saving name prefix', default='', type=str)
    parser.add_argument('--max_epoch', dest='max_epoch',
                        help='max training epochs', default=0, type=int)
    parser.add_argument('--use_extend', dest='use_extend',
                        help='using use extend annots or not', action='store_true')
    parser.add_argument('--no-use_extend', dest='use_extend',
                        help='using use extend annots or not', action='store_false')
    parser.set_defaults(use_extend=True)
    parser.add_argument('--activation', dest='activation',
                        help='sigmoid or softmax for classification', default='', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    cfg.load_from_yaml(args.cfg_file, cfg)

    print('---- config from CML ----')
    if args.restore and os.path.exists(args.restore):
        cfg.restore = args.restore
        print('using {} to restore'.format(args.restore))

    if args.train_dir:
        if not os.path.exists(args.train_dir): os.makedirs(args.train_dir)
        cfg.train_dir = args.train_dir
        print('using {} as train dir'.format(args.train_dir))

    if args.solver:
        cfg.solver = args.solver
        print('using {} as solver'.format(args.solver))

    if args.focal_loss:
        cfg.use_focal_loss = args.focal_loss
        print('using focal loss {}'.format(args.focal_loss))
    else:
        cfg.use_focal_loss = args.focal_loss
        print('using focal loss {}'.format(args.focal_loss))

    if args.lr > 0:
        cfg.lr = args.lr
        print('using learning rate {}'.format(args.lr))

    if args.save_prefix:
        cfg.save_prefix = args.save_prefix
        print('using {} as save prefix'.format(args.save_prefix))

    if args.max_epoch > 0:
        cfg.max_epoch = args.max_epoch
        print('using max epochs of {}'.format(args.max_epoch))

    if args.activation:
        cfg.class_activation = args.activation
        print('using {} as activation function'.format(cfg.class_activation))

    cfg.use_extend = bool(args.use_extend)
    print('using extend annots {}'.format(args.use_extend))

    print('---- config from CML end ----')
    print()

    return args

def log_images(imgs, names, global_step, prefix='Image'):
    summary = []
    for i, I in enumerate(imgs):
        summary.append(tbx.summary.image('%d/%s-%s'%(global_step, prefix, names[i]), I))
    return summary

args = parse_args()


# config model and lr
num_anchors=len(cfg.anchor_ratios) * len(cfg.anchor_scales[0]) * len(cfg.anchor_shift)\
    if isinstance(cfg.anchor_scales[0], list) else \
    len(cfg.anchor_ratios) * len(cfg.anchor_scales)

resnet = resnet50 if cfg.backbone == 'resnet50' else resnet101
detection_model = MaskRCNN if cfg.model_type.lower() == 'maskrcnn' else RetinaNet

model = detection_model(resnet(pretrained=True, frozen=cfg.frozen, maxpool5=cfg.maxpool5),
                        num_classes=cfg.num_classes, num_anchors=num_anchors,
                        strides=cfg.strides, in_channels=cfg.in_channels, f_keys=cfg.f_keys,
                        num_channels=256, is_training=True, activation=cfg.class_activation)

if torch.cuda.device_count() > 1:
    print()
    print('--- using %d gpus ---' % torch.cuda.device_count())
    print()
    model_ori = model
    model = ListDataParallel(model)
    cfg.batch_size = cfg.batch_size * torch.cuda.device_count()
    cfg.data_workers = cfg.data_workers * torch.cuda.device_count()
    cfg.lr = cfg.lr * torch.cuda.device_count()
else:
    model_ori = model

lr = cfg.lr
start_epoch = 0
if cfg.restore is not None:
    print('Restoring from %s ...' % cfg.restore)
    meta = load_net(cfg.restore, model)
    print (meta)
    if meta[0] >= 0 and not cfg.start_over:
        start_epoch = meta[0] + 1
        lr = meta[1]
    print('Restored from %s, starting from %d epoch, lr:%.6f' % (cfg.restore, start_epoch, lr))

trainable_vars = [param for param in model.parameters() if param.requires_grad]

# for k, var in dict(model.named_parameters()).items():
#     if var.requires_grad:
#         print('gradients --- ', k)

if cfg.solver == 'SGD':
    optimizer = torch.optim.SGD(trainable_vars, lr, cfg.momentum, weight_decay=cfg.weight_decay)
elif cfg.solver == 'RMS':
    optimizer = torch.optim.RMSprop(trainable_vars, lr, cfg.momentum, weight_decay=cfg.weight_decay)
elif cfg.solver == 'Adam':
    optimizer = torch.optim.Adam(trainable_vars, lr, weight_decay=cfg.weight_decay)
else:
    optimizer = ''
    raise ValueError('Unknown solver {}'.format(cfg.solver))
model.cuda()
# model.eval()

## data loader
get_loader = get_data_loader(cfg.datasetname)
train_data = get_loader(cfg.data_dir, cfg.split, is_training=True,
                        batch_size=cfg.batch_size, num_workers=cfg.data_workers)
class_names = train_data.dataset.classes
print('dataset len: {}'.format(len(train_data.dataset)))

tb_dir = os.path.join(cfg.train_dir, cfg.backbone + '_' + cfg.datasetname, time.strftime("%h%d_%H"))
writer = tbx.FileWriter(tb_dir)
summary_out = []

global_step = 0
timer = Timer()
for ep in range(start_epoch, cfg.max_epoch):
    if ep in cfg.lr_decay_epoches and cfg.solver == 'SGD':
        lr *= cfg.lr_decay
        adjust_learning_rate(optimizer, lr)
        print ('adjusting learning rate %.6f' % lr)

    for step, batch in enumerate(train_data):
        timer.tic()

        input, anchors_np, im_scale_list, image_ids, gt_boxes_list, rpn_targets, _, _ = batch
        gt_boxes_list = ScatterList(gt_boxes_list)
        input = everything2cuda(input)
        rpn_targets = everything2cuda(rpn_targets)
        outs = model(input, gt_boxes_list, anchors_np, rpn_targets=rpn_targets)

        if cfg.model_type == 'maskrcnn':
            rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts, anchors, \
            rois, roi_img_ids, rcnn_logit, rcnn_box, rcnn_prob, rcnn_labels, rcnn_bbtargets, rcnn_bbwghts = outs
            outputs = [rois, roi_img_ids, rpn_logit, rpn_box, rpn_prob, rcnn_logit, rcnn_box, rcnn_prob, anchors]
            targets = [rpn_labels, rpn_bbtargets, rpn_bbwghts, rcnn_labels, rcnn_bbtargets, rcnn_bbwghts]
        elif cfg.model_type == 'retinanet':
            rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts = outs
            outputs = [rpn_logit, rpn_box, rpn_prob]
            targets = [rpn_labels, rpn_bbtargets, rpn_bbwghts]
        else:
            raise ValueError('Unknown model type: %s' % cfg.model_type)

        loss_dict = model_ori.build_losses(outputs, targets)
        loss = model_ori.loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = timer.toc()

        if step % cfg.display == 0:
            loss_str = ', '.join('%s: %.3f' % (k, v) for k, v in loss_dict.iteritems())
            print(time.strftime("%H:%M:%S ") +
                  'Epoch %d iter %d: speed %.3fs, loss %.3f ' % (ep, step, t, loss.data.cpu().item())
                  + loss_str + ' lr:%.5f' % lr)

            summary_out2 = []
            for k, v in loss_dict.iteritems():
                summary_out2.append(tbx.summary.scalar('Loss/'+k, v.data.cpu().numpy()))
            summary_out2.append(tbx.summary.scalar('LR', lr))
            for s in summary_out2: writer.add_summary(s, float(global_step))

        if step % 5000 == 0 and global_step != 0:
            if not cfg.save_prefix:
                save_path = os.path.join(cfg.train_dir, cfg.model_type + '_' + cfg.backbone + '_' +
                                        cfg.datasetname + '_ep' + str(ep) + '.h5')
            else:
                save_path = os.path.join(cfg.train_dir, cfg.save_prefix + '_ep' + str(ep) + '.h5')
            save_net(save_path, model, epoch=ep, lr=lr)
            print('save to {}'.format(save_path))

        # drawing
        if global_step % cfg.log_image == 0:

            summary_out = []
            input_np = everything2numpy(input)

            # draw detection
            dets_dict = model_ori.get_final_results(outputs, everything2cuda(anchors_np),
                                                    score_threshold=cfg.score_threshold,
                                                    max_dets=cfg.max_det_num,
                                                    overlap_threshold=cfg.overlap_threshold)
            for key, dets in dets_dict.iteritems():
                Is = single_shot.draw_detection(input_np, dets, class_names=class_names)
                Is = Is.astype(np.uint8)
                summary_out += log_images(Is, image_ids, global_step, prefix='Detection_' + key)

            # draw gt
            Is = single_shot.draw_gtboxes(input_np, gt_boxes_list, class_names=class_names)
            Is = Is.astype(np.uint8)
            summary_out += log_images(Is, image_ids, global_step, prefix='GT')

            # # draw positive anchors on images
            # if True:
            #     Imgs, cnt = single_shot.draw_anchors(everything2numpy(input), everything2numpy(rpn_targets),
            #                                          anchors_np, class_names=class_names)
            #     Imgs = Imgs.astype(np.uint8)
            #     summary_out += log_images(Imgs, image_ids, global_step, prefix='GT_anchor')
            #
            #     print (time.strftime("%H:%M:%S ") + '{} positive anchors'.format(cnt))

            summary = model_ori.get_summaries(is_training=True)
            for s in summary: writer.add_summary(s, float(global_step))
            for s in summary_out: writer.add_summary(s, float(global_step))
            summary_out = []

        global_step += 1

    if not cfg.save_prefix:
        save_path = os.path.join(cfg.train_dir, cfg.model_type + '_' + cfg.backbone + '_' +
                                 cfg.datasetname + '_ep' + str(ep) + '.h5')
    else:
        save_path = os.path.join(cfg.train_dir, cfg.save_prefix + '_ep' + str(ep) + '.h5')
    save_net(save_path, model, epoch=ep, lr=lr, log=False)
    print('save to {}'.format(save_path))

writer.close()
