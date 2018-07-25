#!/usr/bin/env python
# coding=utf-8
# This file is copied from torchvision.models
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math
import os
import torch
import torch.utils.model_zoo as model_zoo
from . import utils



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 1 if dilation == 1 else 2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.maxpool5 = kwargs['maxpool5'] if 'maxpool5' in kwargs else True
        if self.maxpool5:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        else:
            self.layer4 = self._make_layer_no_downsample(block, 512, layers[3], stride=2)
            print('removing subsample 5 ... using dilation')

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if 'frozen' in kwargs:
            self.frozen(stage=kwargs['frozen'])

        # using a small kernel to conv image 3x3 instead of 7x7
        self.conv1_s = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.layer1_retain = self._make_layer1_retain(block, 64, layers[0], use_dilation=True)
        self.layer1_retain = self._make_layer1_retain(block, 64, layers[0], use_dilation=False)

    def de_frozen(self):

        for p in self.conv1.parameters(): p.requires_grad = True
        for p in self.bn1.parameters(): p.requires_grad = True
        for p in self.layer2.parameters(): p.requires_grad = True
        for p in self.layer3.parameters(): p.requires_grad = True
        for p in self.layer4.parameters(): p.requires_grad = True

    def frozen(self, stage=2):
        def set_bn_fixed(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.apply(set_bn_fixed)
        self.apply(set_bn_eval)
        if stage >= 1:
            for p in self.conv1.parameters(): p.requires_grad = False
            for p in self.bn1.parameters(): p.requires_grad = False
            #if hasattr(self, 'conv1_s'):
            #    for p in self.conv1_s.parameters(): p.requires_grad = False
        if stage >= 2:
            for p in self.layer1.parameters(): p.requires_grad = False
            # if hasattr(self, 'layer1_retain'):
            #     for p in self.layer1_retain.parameters(): p.requires_grad = False
        if stage >= 3:
            for p in self.layer2.parameters(): p.requires_grad = False
        if stage >= 4:
            for p in self.layer3.parameters(): p.requires_grad = False
        if stage >= 5:
            for p in self.layer4.parameters(): p.requires_grad = False

        print('backbone: frozen stage 0 ~ %d' % stage)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_layer_no_downsample(self, block, planes, blocks, stride=1, dilation=1):
        """replace downsample with dilation conv, keep the same spatial dimensions,
        NOTE: only replace the first 3x3 conv to dilation conv with stride=1
        """
        downsample = None
        # assert stride == 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, 1, downsample, dilation=stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_layer1_retain(self, block, planes, blocks, stride=1, dilation=1, use_dilation=False):
        """conduct conv with dilation conv, share same parameters with layer1"""
        layers = []
        downsample = nn.Sequential(
            nn.Conv2d(64, planes * block.expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
        if use_dilation:
            layers.append(block(64, planes, 1, downsample, dilation=2))
        else:
            layers.append(block(64, planes, 1, downsample, dilation=1))
        for i in range(1, blocks):
            layers.append(block(64 * 4, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _share_param_layer1(self):

        if not hasattr(self, 'layer1_retain'):
            return

        print('loading layer1 into layer1_retain ...')
        layer1 = dict(self.layer1.named_parameters())
        layer1_retain = dict(self.layer1_retain.named_parameters())
        # print(sorted(layer1.keys()))
        # print(sorted(layer1_retain.keys()))
        for k, var in layer1_retain.items():
            # print(var.size(), '<---', layer1[k].size())
            var.requires_grad = False
            var.copy_(layer1[k])

        layer1 = dict(self.layer1.named_parameters())
        layer1_retain = dict(self.layer1_retain.named_parameters())
        for k, var in layer1_retain.items():
            v1, v2 = var, layer1[k]
            # print(v1.size(), '<-->', v2.size())
            v = v1 - v2
            assert -1e-6 < v.max() < 1e-6 and -1e-6 < v.min() < 1e-6

        print('loading layer1 into layer1_retain ... done')

    def _share_param_conv1(self):

        if not hasattr(self, 'conv1_s'):
            print('no conv1_s layer')
            return

        print('loading conv1 into conv1_s ...')
        w = self.conv1.weight
        # out, size, size, in
        w = w.permute(0, 2, 3, 1)
        w_np = w.data.numpy()

        import cv2
        import numpy as np
        import os
        ws = []
        cnt = 0
        for w_ in w_np:
            im = 255 * (w_ - w_.min()) / (w_.max() - w_.min())
            im = im.astype(np.uint8)
            r, g, b = cv2.split(im)
            im = cv2.merge((b, g, r))
            im = cv2.resize(im, (70, 70), interpolation=cv2.INTER_NEAREST)
            if os.path.exists('output/kernels/'):
                cv2.imwrite('output/kernels/%d.jpg' % cnt, im)
            cnt += 1
            w_ = cv2.resize(w_, (5, 5))
            w_ = cv2.resize(w_, (3, 3))
            # ws.append(cv2.resize(w_, (3, 3)))
            ws.append(w_)

        ws = np.stack(ws)
        # out, in, size, size
        ws = ws.transpose(0, 3, 1, 2)
        ws = torch.tensor(ws)

        self.conv1_s.weight.data.copy_(ws)
        print('loading conv1 into conv1_s ... done')

    def forward(self, x):

        endpoints = {}
        endpoints['C0'] = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        endpoints['C1'] = x
        x = self.maxpool(x)
        x = self.layer1(x)
        endpoints['C2'] = x
        x = self.layer2(x)
        endpoints['C3'] = x
        x = self.layer3(x)
        endpoints['C4'] = x
        x = self.layer4(x)
        endpoints['C5'] = x

        return endpoints

def resnet50(pretrained=False, weight_path=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        model._share_param_layer1()
        model._share_param_conv1()
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        model._share_param_layer1()
        model._share_param_conv1()
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
