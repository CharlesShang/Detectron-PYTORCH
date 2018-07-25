from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pascal_voc import get_loader as pascal_dataloader
from .coco2 import get_loader as coco_dataloader
from .citypersons2 import get_loader as citypersons_dataloader
from .image_folder import get_loader as image_folder_dataloader
import libs.configs.config as cfg


def get_data_loader(datasetname=None):
    if datasetname == None:
        datasetname = cfg.datasetname

    if datasetname == 'coco':
        return coco_dataloader
    elif datasetname == 'pascal_voc':
        return pascal_dataloader
    elif datasetname == 'citypersons':
        return citypersons_dataloader
    elif datasetname == 'image_folder':
        return image_folder_dataloader
    else:
        raise ValueError('{} is not supported'.format(datasetname))
