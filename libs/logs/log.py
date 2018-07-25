from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import libs.configs.config as cfg

def LOG(mssg, onscreen=True):
    if not os.path.exists(cfg.train_dir):
        os.makedirs(cfg.train_dir)
    logging.basicConfig(filename=cfg.train_dir + '/maskrcnn.log',
                      level=logging.INFO,
                      datefmt='%m/%d/%Y %I:%M:%S %p', format='%(asctime)s %(message)s')
    logging.info(mssg)
    if onscreen:
        print (mssg)