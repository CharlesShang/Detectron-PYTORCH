#!/usr/bin/env python
# coding=utf-8
# This file is copied from torchvision.models
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py
import math

class Xt(nn.Module):
    """from ResneXt"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, relu=True, same_padding=True, bias=True,
                 paths=32, hiddens=4, shortcut=True):
        super(Xt, self).__init__()
        self.paths = paths
        self.shortcut = shortcut
        assert shortcut and in_channels == out_channels or not shortcut, \
            "if shotcut is set, in and out channels must be equal {} vs {}".format(in_channels, out_channels)

        self.params = []
        for i in range(paths):
            conv1 = Conv2d_BatchNorm(in_channels, hiddens, 1, 1, same_padding=True, bias=bias)
            setattr(self, 'reduce_conv%d' % i, conv1)
            conv2 = Conv2d_BatchNorm(hiddens, hiddens, kernel_size, stride, same_padding=same_padding, bias=bias)
            setattr(self, 'trans_conv%d' % i, conv2)
            conv3 = Conv2d_BatchNorm(hiddens, out_channels, 1, 1, relu=relu, same_padding=True, bias=bias)
            setattr(self, 'increase_conv%d' % i, conv3)
            self.params.append([conv1, conv2, conv3])

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)

    def forward(self, x):
        f = []
        s = None
        for p in self.params:
            x_ = p[0](x)
            x_ = p[1](x_)
            x_ = p[2](x_)
            s = s + x_ if s is not None else x_

        s = self.bn(s + x)
        return s

class Conv2dBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, same_padding=True, bias=True):
        super(Conv2dBottleNeck, self).__init__()
        channels = out_channels // 4
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv_1x1_reduce = nn.Conv2d(in_channels, channels, kernel_size=1, padding=0, bias=bias, stride=1)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=bias, stride=stride)
        self.conv_1x1 = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0, bias=bias, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1x1_reduce(x)
        x = self.relu(x)
        x = self.conv_3x3(x)
        x = self.relu(x)
        x = self.conv_1x1(x)
        x = self.relu(x)
        return x

class Conv2dBottleNeck_BatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, same_padding=True, bias=False, momentum=0.1):
        super(Conv2dBottleNeck_BatchNorm, self).__init__()
        channels = out_channels // 4
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv_1x1_reduce = nn.Conv2d(in_channels, channels, kernel_size=1, padding=0, bias=bias, stride=1)
        self.bn_1x1_reduce = nn.BatchNorm2d(channels, momentum=momentum)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, bias=bias, stride=stride)
        self.bn_3x3 = nn.BatchNorm2d(channels, momentum=momentum)
        self.conv_1x1 = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0, bias=bias, stride=1)
        self.bn_1x1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1x1_reduce(x)
        x = self.bn_1x1_reduce(x)
        x = self.relu(x)
        x = self.conv_3x3(x)
        x = self.bn_3x3(x)
        x = self.relu(x)
        x = self.conv_1x1(x)
        x = self.bn_1x1(x)
        x = self.relu(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bias=False, momentum=0.01):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        # momentum = 0.05 if self.training else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=True) if relu else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv2d_Transpose_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, relu=True, same_padding=False, bias=False):
        super(Conv2d_Transpose_BatchNorm, self).__init__()
        padding = int((kernel_size - stride) / 2) if same_padding else 0

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv2d_Transpose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, relu=True, same_padding=False, bias=False):
        super(Conv2d_Transpose, self).__init__()
        padding = int((kernel_size - stride) / 2) if same_padding else 0

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def init_conv_weight(module, std=0.001):
    d = dict(module.named_parameters())
    if 'conv.weight' in d.keys():
        nn.init.normal(d['conv.weight'], std=std)
    if 'conv.bias' in d.keys():
        nn.init.constant(d['conv.bias'], 0)

def init_new_layers(module, std=0.01):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(1. / n))
            # m.weight.data.normal_(0, std)
            try:
                m.bias.data.zero_()
            except:
                print ('has no bias')
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, std)
            try:
                m.bias.data.zero_()
            except:
                print ('has no bias')


def init_guass(module, std):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(0, std)
            try:
                m.bias.data.zero_()
            except:
                print('has no bias')


def init_xavier(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            print(m.weight.size(), 'xavier init')
            try:
                m.bias.data.zero_()
            except:
                print('has no bias')


def init_msra(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform(m.weight)
            print(m.weight.size(), 'xavier init')
            try:
                m.bias.data.zero_()
            except:
                print('has no bias')


def everything2cuda(x, volatile=False):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).cuda()
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor) or\
        isinstance(x, torch.IntTensor) or isinstance(x, torch.DoubleTensor):
        return x.cuda()
    elif isinstance(x, Variable):
        return x.cuda()
    elif isinstance(x, list) or isinstance(x, tuple):
        y = []
        for i, e in enumerate(x):
            y.append(everything2cuda(e))
        return y
    else:
        print ('Unknown data type', type(x))
        raise TypeError('Unknown data type')
    

def everything2tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, Variable):
        if x.is_cuda:
            return x.cpu().data
        return x.data
    elif isinstance(x, list) or isinstance(x, tuple):
        y = []
        for i, e in enumerate(x):
            y.append(everything2tensor(e))
        return y
    else:
        print ('Unknown data type')
        raise TypeError

def everything2numpy(x):
    if isinstance(x, torch.FloatTensor) or \
          isinstance(x, torch.IntTensor) or \
          isinstance(x, torch.DoubleTensor) or \
          isinstance(x, torch.LongTensor):
        return x.numpy().copy()
    if isinstance(x, Variable):
        if x.is_cuda:
            return x.cpu().data.numpy()
        return x.data.numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        y = []
        for i, e in enumerate(x):
            y.append(tensor2numpy(e))
        return y
    else:
        print ('Unknown data type')
        raise TypeError

tensor2numpy = everything2numpy

def everything2cpu(x):
    if isinstance(x, np.ndarray):
        return Variable(torch.from_numpy(x), requires_grad=False).cpu()
    elif isinstance(x, Variable):
        return x.cpu()
    elif isinstance(x, list) or isinstance(x, tuple):
        y = []
        for i, e in enumerate(x):
            y.append(everything2cpu(e))
        return y
    else:
        print ('Unknown data type')
        raise TypeError


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_net(fname, net, epoch=-1, lr=-1, log=False):
    with h5py.File(fname, mode='w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
            if log:
                print('%s, mean: %.4f, std: %.4f' % (k, v.mean(), v.std()))
                print('SAVE: {} {}'.format(k, v.size()))

        h5f.attrs['epoch'] = epoch
        h5f.attrs['lr'] = lr


def load_net(fname, net, force=True, log=False):
    with h5py.File(fname, mode='r') as h5f:
        loaded = {k: False for k in h5f.keys()}
        for k, v in net.state_dict().items():
            if k not in h5f:
                k = 'module.' + k
            if k in h5f:
                loaded[k] = True
                param = torch.from_numpy(np.asarray(h5f[k]))
                if v.size() == param.size():
                    v.copy_(param)
                    if log:
                        print('%s, mean: %.4f, std: %.4f' % (k, param.mean(), param.std()))
                        print('LOAD: loaded {} {}'.format(k, v.size()))
                elif force:
                    print('LOAD: ignoring layer: {} {} vs {}'.format(k, param.size(), v.size()))
                    if k.startswith('rpns'):
                        load_coco_cls_layer_to_detrac_sigmoid(param, v)
                else:
                    raise ValueError('LOAD: tensors sizes dont match:', param.size(), ' vs ', v.size())
            else:
                print('LOAD: ignoring layer: {}'.format(k))

        epoch = h5f.attrs['epoch'] if 'epoch' in h5f.attrs else -1
        lr = h5f.attrs['lr'] if 'lr' in h5f.attrs else -1.
        for k in loaded.keys():
            if loaded[k] == False:
                print ('LOAD: xxxx {} has not been loaded xxxx'.format(k))

        return epoch, lr


def load_coco_cls_layer_to_detrac_sigmoid(src, dest):
    num_anchors = 10
    coco_classes = 80
    detrac_classes = 4
    order_map = {1:3, 2:3, 3:6, 4:3} # from detrac to coco
    for t in order_map:
        k = order_map[t]
        # for i in range(10):
        #     dest[i * 4 + t - 1] = src[i * 80 + k - 1]
        print((k - 1) * 10, '--', k * 10, '---->', (t - 1) * 10, '--', t * 10)
        dest[(t - 1)::detrac_classes] = src[(k - 1)::coco_classes]

def load_coco_cls_layer_to_detrac_softmax(src, dest):
    num_anchors = 10
    coco_classes = 80
    detrac_classes = 4
    order_map = {1: 3, 2: 3, 3: 6, 4: 3}  # from detrac to coco
    for t in order_map:
        k = order_map[t]
        # for i in range(10):
        #     dest[i * 4 + t - 1] = src[i * 80 + k - 1]
        print((k - 1) * 10, '--', k * 10, '---->', (t - 1) * 10, '--', t * 10)
        dest[(t - 1)::detrac_classes] = src[(k - 1)::coco_classes]