import cPickle
import os, sys
import zipfile
from six.moves import urllib

import cv2
import ujson as json
import numpy as np

import torch
import torch.utils.data as data

from libs.logs.log import LOG
from dataloader import sDataLoader, mkdir
from libs.datasets.pycocotools.coco import COCO
from libs.datasets.pycocotools.cocoeval import COCOeval
import libs.configs.config as cfg
from libs.nets.utils import everything2tensor, everything2cuda
from libs.boxes.anchor import anchors_plane, anchor_pyramid
from libs.layers.data_layer import data_layer_keep_aspect_ratio, \
    data_layer_keep_aspect_ratio_batch

_TRAIN_DATA_URL="https://msvocds.blob.core.windows.net/coco2014/train2014.zip"
_VAL_DATA_URL="https://msvocds.blob.core.windows.net/coco2014/val2014.zip"
_INS_LABEL_URL="https://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip"
_KPT_LABEL_URL="https://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip"
_CPT_LABEL_URL="https://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
_DATA_URLS=[
  _TRAIN_DATA_URL, _VAL_DATA_URL,
  _INS_LABEL_URL, _KPT_LABEL_URL, _CPT_LABEL_URL,
]

def download_and_uncompress_zip(zip_url, dataset_dir):
  """Downloads the `zip_url` and uncompresses it locally.
     From: https://github.com/tensorflow/models/blob/master/slim/datasets/dataset_utils.py

  Args:
    zip_url: The URL of a zip file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = zip_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

  if os.path.exists(filepath):
    print('Zip file already exist. Skip download..', filepath)
  else:
    filepath, _ = urllib.request.urlretrieve(zip_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  with zipfile.ZipFile(filepath) as f:
    print('Extracting ', filepath)
    f.extractall(dataset_dir)
    print('Successfully extracted')

class coco(data.Dataset):

    classes = ('')

    def __init__(self, data_dir, split, is_training):
        self._split = split
        self._data_dir = data_dir
        self._data_size = -1
        self._is_training = is_training
        self.classes = []

        print ('loading coco ... ')
        if self._split in ['train2014', 'val2014', 'minival2014', 'minival2014_new']:
            self._load()
        elif self._split in ['trainval2014',]:
            self._load_trainvalsplit()
        else:
            raise ValueError('Check your datasetting: {}' .format(self._split))

        # if self._is_training:
        #     print ('building label balance ...')
        #     self.label_balance()
        #     print ('building label done')

    def _load(self, ):
        assert self._split in ['train2014', 'val2014', 'minival2014', 'minival2014_new']

        annFile = os.path.join(self._data_dir, 'annotations', 'instances_%s.json' % (self._split))
        coco = COCO(annFile)
        # imgs = [(img_id, coco.imgs[img_id]) for img_id in coco.imgs]
        imgs = []
        for img_id in coco.imgs:
            if img_id == 320612:
                continue
            imgs.append((img_id, coco.imgs[img_id], 0))

        self._data_size = len(imgs)
        self._imgs = imgs
        self._cocos = (coco, )
        self.classes = [u'background'] + [cls['name'] for cls in coco.loadCats(coco.getCatIds())]

        return

    def _load_trainvalsplit(self, ):
        assert self._split in ['trainval2014']

        annotation_dir = os.path.join(self._data_dir, 'annotations')
        annFile = os.path.join(annotation_dir, 'instances_trainval2014.json')
        coco_train = COCO(annFile)

        imgs = [(img_id, coco_train.imgs[img_id], 0) for img_id in coco_train.imgs if img_id != 320612]

        self._data_size = len(imgs)
        self._imgs = imgs
        self._cocos = (coco_train, )
        self.classes = [u'background'] + [cls['name'] for cls in coco_train.loadCats(coco_train.getCatIds())]

        return

    def _is_in_minival(self, img_id):
        for img in self._minival['images']:
            if (img['id']) == (img_id):
                return True
        return False

    def _real_id_to_cat_id(self, catId):
        """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
        real_id_to_cat_id = \
            {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16,
             16: 17,
             17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33,
             30: 34,
             31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48,
             44: 49,
             45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62,
             58: 63,
             59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80,
             72: 81,
             73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}
        return real_id_to_cat_id[catId]


    def _cat_id_to_real_id(self, readId):
        """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
        cat_id_to_real_id = \
            {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15,
             17: 16,
             18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29,
             34: 30,
             35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43,
             49: 44,
             50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57,
             63: 58,
             64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71,
             81: 72,
             82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
        return cat_id_to_real_id[readId]

    def _get_coco_masks(self, coco, img_id, height, width, img_name):
        """ get the masks for all the instances
        Note: some images are not annotated
        Return:
          masks, mxhxw numpy array
          classes, mx1
          bboxes, mx4
        """
        annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        # assert  annIds is not None and annIds > 0, 'No annotaion for %s' % str(img_id)
        anns = coco.loadAnns(annIds)
        # assert len(anns) > 0, 'No annotaion for %s' % str(img_id)
        masks = []
        classes = []
        bboxes = []
        mask = np.zeros((height, width), dtype=np.float32)
        segmentations = []
        for ann in anns:
            m = coco.annToMask(ann)  # zero one mask
            # m = np.zeros([height, width], dtype=np.int32)
            assert m.shape[0] == height and m.shape[1] == width, \
                'image %s and ann %s dont match' % (img_id, ann)
            masks.append(m)
            cat_id = self._cat_id_to_real_id(ann['category_id'])
            if ann['iscrowd']:
                cat_id = -1
            classes.append(cat_id)
            bboxes.append(ann['bbox'])
            m = m.astype(np.float32) * cat_id
            mask[m > 0] = m[m > 0]

        masks = np.asarray(masks)
        classes = np.asarray(classes)
        bboxes = np.asarray(bboxes)
        # to x1, y1, x2, y2
        if bboxes.shape[0] <= 0:
            bboxes = np.zeros([0, 4], dtype=np.float32)
            classes = np.zeros([0], dtype=np.float32)
            masks = np.zeros([1, height, width], dtype=np.int32) # force to 1
            # print ('None Annotations %s' % img_name)
            LOG('None Annotations %s' % img_name, onscreen=False)
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        assert classes.shape[0] == bboxes.shape[0], 'Processing Annotation Error'

        return bboxes.astype(np.float32), classes.astype(np.float32), masks.astype(np.int32), mask.astype(np.int32)

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, i):

        img_id = self._imgs[i][0]
        img_name = self._imgs[i][1]['file_name']
        split = img_name.split('_')[1]
        img_name = os.path.join(self._data_dir, split, img_name)

        coco = self._cocos[self._imgs[i][2]]
        # if self._split == 'trainval2014' and not self._imgs[i][2]:
        #     coco = self._cocos[1]

        height, width = self._imgs[i][1]['height'], self._imgs[i][1]['width']
        bboxes, classes, inst_masks, mask = self._get_coco_masks(coco, img_id, height, width, img_name)
        # img = np.array(Image.open(img_name))

        # do some preprocessings here
        im, im_scale, annots = data_layer_keep_aspect_ratio(img_name, bboxes, classes, inst_masks, mask,
                                                            self._is_training)
        # im = np.transpose(im, [2, 0, 1]) # c, h, w

        return im, im_scale, annots, img_id

    def to_detection_format(self, Dets, image_ids, im_scale_list):
        """Add a detection results to list"""
        dict_list = []
        assert len(Dets) == len(im_scale_list)
        for i, (dets, img_id) in enumerate(zip(Dets, image_ids)):
            for box in dets:
                if im_scale_list is not None:
                    scale = im_scale_list[i]
                    box[0:4] = box[0:4] / scale
                x, y = box[0], box[1]
                width, height = box[2] - box[0], box[3] - box[1]
                score, id = box[4], int(box[5])
                id = self._real_id_to_cat_id(id)
                dict = {
                    "image_id": img_id,
                    "category_id": id,
                    "bbox": [round(x, 1), round(y, 1), round(width, 1), round(height, 1)],
                    "score": round(score, 3)
                }
                dict_list.append(dict)
        return dict_list

    def eval(self, result_file):
        """evaluation detection results"""
        annType = ['segm', 'bbox', 'keypoints']
        annType = annType[1]  # specify type here
        assert self._split in ['val2014', 'minival2014', 'minival2014_new']
        cocoGt = self._cocos[0]
        cocoDt = cocoGt.loadRes(result_file)
        cocoEval = COCOeval(cocoGt, cocoDt, annType)

        cocoEval.params.useCats = 1
        cocoEval.evaluate()
        cocoEval.accumulate()
        print ('With ALL CatIds...')
        cocoEval.summarize()

        cocoEval.params.useCats = 0
        cocoEval.evaluate()
        cocoEval.accumulate()
        print ('Without CatIds...')
        cocoEval.summarize()

        cocoEval.params.useCats = 1
        for i in range(1, 81):
            catId = self._real_id_to_cat_id(i)
            cocoEval.params.catIds = [catId]
            cocoEval.evaluate()
            cocoEval.accumulate()
            print ('With CatIds {} {}'.format(catId, cocoGt.cats[catId]['name']))
            cocoEval.summarize()

    def label_balance(self, alpha=50):
        """since coco is extramely unbalanced, here we do data augmentation to balance among classes"""
        _label_to_images = {}
        _image_to_labels = {}
        coco = self._cocos[0]
        annIds = coco.getAnnIds(iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if ann['iscrowd']:
                continue
            img_id = ann['image_id']
            cat_id = ann['category_id']
            if not _label_to_images.has_key(cat_id):
                _label_to_images[cat_id] = [img_id]
            else:
                _label_to_images[cat_id].append(img_id)

            if not _image_to_labels.has_key(img_id):
                _image_to_labels[img_id] = [cat_id]
            else:
                _image_to_labels[img_id].append(cat_id)

        max_label = ''
        max_num = 0
        for k in _label_to_images:
            if max_num < len(_label_to_images[k]):
                max_num = len(_label_to_images[k])
                max_label = k
        print ('max label is : %s, %d' % (max_label, max_num))

        all_image_ids = coco.getImgIds()
        for cat_id in _label_to_images:
            sample_times = max_num // (alpha * len(_label_to_images[cat_id]))
            sample_times = max(1, sample_times)
            for img_id in _label_to_images[cat_id]:
                if sample_times > 1:
                    all_image_ids.extend([img_id] * (sample_times - 1))

        _imgs = []
        for img_id in all_image_ids:
            if img_id == 320612:
                continue
            _imgs.append((img_id, coco.imgs[img_id], 0))
        print ('add data from %d to %d' % (len(self._imgs), len(_imgs)))
        self._imgs = _imgs


def collate_fn(data):
    # print ('collate_fn:', len(data))
    im_batch, im_scale_batch, anchors, rpn_targets, inst_masks_batch, mask_batch = \
        data_layer_keep_aspect_ratio_batch(data, is_training=True)

    input = torch.stack(everything2tensor(im_batch))
    # input = np.stack(im_batch)

    gt_boxes_list = [d[2]['gt_boxes'] for d in data]
    image_ids = [d[3] for d in data]
    im_scale_list = [d[1] for d in data]

    labels = torch.stack(everything2tensor([d[2]['rpn_targets'][0] for d in data]))
    # label_weights = torch.stack(everything2tensor(rpn_targets['labels_weights_batch']))
    bbox_targets = torch.stack(everything2tensor([d[2]['rpn_targets'][1] for d in data]))
    bbox_inside_weights = torch.stack(everything2tensor([d[2]['rpn_targets'][2] for d in data]))

    rpn_targets = [labels, bbox_targets, bbox_inside_weights]
    mask = torch.stack(everything2tensor(mask_batch))
    inst_masks = everything2tensor(inst_masks_batch)  # a list of (N, H, W). N can be changable

    return input, anchors, im_scale_list, image_ids, gt_boxes_list, rpn_targets, inst_masks, mask


def collate_fn_testing(data):

    im_batch, im_scale_batch, anchors, _, _, _ = \
        data_layer_keep_aspect_ratio_batch(data, is_training=False)

    input = torch.stack(everything2tensor(im_batch))
    gt_boxes_list = [d[2]['gt_boxes'] for d in data]
    im_scale_list = [d[1] for d in data]
    image_ids = [d[3] for d in data]

    return input, anchors, im_scale_list, image_ids, gt_boxes_list


def get_loader(data_dir, split, is_training, batch_size=16, shuffle=True, num_workers=4):
    dataset = coco(data_dir, split, is_training)
    if is_training:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)
    else:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn_testing)


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from libs.visualization.vis import draw_detection
    from libs.nets.utils import tensor2numpy
    from libs.utils.timer import Timer

    ############### without dataloader
    if False:
        dset = coco('/home/shang/repos/maskrcnn/data/coco', 'val2014', True)
        t = Timer()
        t.tic()
        cnt = 0
        for _ in range(10):
            data = []
            for _ in range(6):
                d = dset[cnt]
                cnt += 1
                data.append(d)
            batch = collate_fn(data)
        end_t = t.toc() / 10.0
        print ('Time:', end_t)

    # loader = get_loader('/home/shang/repos/maskrcnn/data/coco', 'val2014', data_layer, True)
    loader = get_loader('/home/shang/repos/maskrcnn/data/coco', 'minival2014', data_layer, True)

    def _chw2hwc(x):
        return np.transpose(x, [1, 2, 0])

    ############### without dataloader
    for j, batch in enumerate(loader):
        input, Targets, inst_masks, masks, downsampled_masks, gt_boxes, image_ids = batch
        input, Targets, inst_masks, masks, downsampled_masks = \
            tensor2numpy([input, Targets, inst_masks, masks, downsampled_masks])
        print input.shape, Targets[0].shape, Targets[1].shape, masks.shape
        n, c, h, w = input.shape
        print (downsampled_masks[0].shape)
        for i in range(n):
            im = (_chw2hwc(input[i]) + 1) * 128
            im = im.astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im2cv = draw_detection(im, gt_boxes[i][:, :4], cls_inds=gt_boxes[i][:, 4])
            cv2.imshow('test', im2cv)
            cv2.imshow('mask', (masks[i] * 255.0 / max(np.max(masks[i]), 1)).astype(np.uint8))
            cv2.imshow('dmask', (downsampled_masks[0][i] * 255.0 / max(np.max(downsampled_masks[0][i]), 1)).astype(np.uint8))
            cv2.waitKey(10)
            m = inst_masks[i].shape[0]
            for j in range(m):
                M = inst_masks[i][j] * 255
                cv2.imshow('mask%d' % j, M.astype(np.uint8))
                cv2.waitKey(0)
