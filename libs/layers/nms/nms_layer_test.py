from __future__ import print_function
from __future__ import absolute_import

import torch
import numpy as np
from torch.autograd import Variable
from .sample_layer import SampleLayer
import torch.nn as nn
n = 10
c = 5
labels = np.random.randint(-1, c, [n]).astype(np.int32)
probs = np.random.rand(n, c).astype(np.float32)
probs[:, 0] = probs[:, 0] + 5
probs = probs / probs.sum(axis=1)[:, np.newaxis]

labels = [-1, 0, 0, 1, 1, 2]
probs = [
    [0.33, 0.33, 0.33], # invalid
    [0.99, 0.01, 0.00], # easy negative
    [0.01, 0.90, 0.09], # hard negative
    [0.20, 0.80, 0.00], # easy positive
    [0.01, 0.00, 0.99], # hard positive
    [0.00, 0.50, 0.50], # midiorce positive
]
# [0, 1, 0, 1, 0, 1]
# cuda [0 1 0 1 0 1]
# cpu [0 1 0 1 0 1]
probs = np.asarray(probs, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int32)
n = 6
c = 3
print (probs)
print (labels)
sample = SampleLayer(num_classes=c)

labels = Variable(torch.from_numpy(labels))
probs = Variable(torch.from_numpy(probs))
labels = labels.cuda()
probs = probs.cuda()
chozens = sample(labels, probs)
print (chozens.data.cpu().numpy())
print (sample.prob_mean.cpu().numpy())
print (sample.prob_var.cpu().numpy())