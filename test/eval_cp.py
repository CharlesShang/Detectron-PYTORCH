#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from libs.datasets.factory import get_data_loader

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json', dest='json',
                        help='citerpersons json file', default='', type=str)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return args

args = parse_args()

if __name__ == '__main__':
    get_loader = get_data_loader('citypersons')
    test_data = get_loader('./data/citypersons', 'val',
                           is_training=False, batch_size=1,
                           num_workers=1, shuffle=False)
    dataset = test_data.dataset
    if not os.path.exists(args.json):
        print('%s do not exist' % args.json)
    dataset.eval_over_scales(args.json)