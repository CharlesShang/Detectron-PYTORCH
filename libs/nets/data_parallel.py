#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import torch
from torch.nn import DataParallel
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter, Gather
from math import ceil


class ScatterList(list):
    pass


class ListDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return tnn_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return tnn_gather(outputs, output_device, dim=self.dim)

def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            # print('var')
            return Scatter.apply(target_gpus, None, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, ScatterList):
            # print('target_gpus:', target_gpus, 'obj:', len(obj))
            # assert len(obj) == len(target_gpus)
            chunk_size = int(ceil(float(len(obj)) / float(len(target_gpus))))
            # print('scatterlist')
            # print (chunk_size, len(obj))
            return [obj[i*chunk_size: (i+1)*chunk_size] for i in range(len(target_gpus))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        # print('others')
        return [obj for targets in target_gpus]

    return scatter_map(inputs)


def tnn_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def tnn_gather(outputs, target_device, dim=0):
    r"""
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        if isinstance(outputs, Variable):
            if target_device == -1:
                return outputs.cpu()
            return outputs.cuda(target_device)

        out = outputs[0]
        if isinstance(out, Variable):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None

        if isinstance(out, ScatterList):
            return tuple(map(gather_map, itertools.chain(*outputs)))

        return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)