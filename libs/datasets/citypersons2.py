from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import scipy.io as sio
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import cPickle
import torch
import time
import json as json

from libs.datasets.dataloader import sDataLoader
import libs.configs.config as cfg
import libs.boxes.cython_bbox as cython_bbox
from libs.nets.utils import everything2tensor
from libs.datasets.cityscapesscripts.helpers.annotation import Annotation
from libs.datasets.cityscapesscripts.helpers.labels import name2label
from libs.datasets.citypersons_eval import COCO, COCOeval
from libs.layers.data_layer import data_layer_keep_aspect_ratio, \
    data_layer_keep_aspect_ratio_batch

"""
dir structure:
./data/
        data/citypersons
        data/citypersons/cityscape
            data/citypersons/cityscape/leftImg8bit/{train|val|test}
            data/citypersons/cityscape/gtFine/{train|val|test}
"""
class citypersons_extend(object):
    block_threshold = 0.3
    block_threshold_low = 0.05

    def __init__(self, data_root, split, is_training=True):
        assert split in ['train', 'val', 'test'], \
            'unknow citypersons split settings: {}'.format(split)

        self.data_root = data_root
        self.annot_path = os.path.join(data_root, 'annotations', 'anno_' + split + '.mat')
        assert os.path.exists(self.annot_path), \
            '{} not found'.format(self.annot_path)

        self.cityscape_path = os.path.join(data_root, 'cityscape')
        self.split = split
        self.extend_dir = os.path.join(self.data_root, 'extend', self.split)
        self.vis_dir = os.path.join(self.data_root, 'visualization', self.split)
        if not os.path.exists(self.extend_dir):
            os.makedirs(self.extend_dir)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        self.gt_annots = []
        self.extend_annots = []
        self.classes = ('__background__', # always index 0
                         'pedestrian', 'rider', 'sitting', 'unusual', 'group')
        # self.classes = ('__background__', # always index 0
        #                 'pedestrian')

        self._is_training = is_training
        self.patch_path = os.path.join(data_root, 'patches')
        if not os.path.exists(self.patch_path):
            os.makedirs(self.patch_path)

        self.all_instances = []
        self.load_gt()
        # self._build_p_anchors()

    def load_gt(self):
        annots = sio.loadmat(self.annot_path)
        annots = annots['anno_'+self.split+'_aligned'].reshape([-1])
        for ann in annots:
            ann = ann.reshape([-1])
            city_name, image_name, bbs = ann[0][0][0], ann[0][1][0], ann[0][2]
            city_name = 'tubingen' if city_name == 'tuebingen' else city_name
            json_name = image_name.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            img_name = os.path.join(self.cityscape_path, 'leftImg8bit', self.split, city_name, image_name)
            json_name = os.path.join(self.cityscape_path, 'gtFine', self.split, city_name, json_name)

            if not os.path.exists(img_name):
                raise ValueError('image {} not found'.format(img_name))

            bbs = bbs.astype(np.float32)
            gt_classes, gt_boxes = bbs[:, 0], bbs[:, 1:5]
            gt_boxes_vis = bbs[:, 6:10]
            gt_boxes[:, 2:4] += gt_boxes[:, 0:2]
            gt_boxes_vis[:, 2:4] += gt_boxes_vis[:, 0:2]
            self.gt_annots.append({
                'img_name': img_name,
                'json_name': json_name,
                'gt_boxes': gt_boxes.astype(np.float32),
                'gt_classes': gt_classes,
                'gt_boxes_vis': gt_boxes_vis.astype(np.float32),
            })


    def get_inst_mask(self, gt_annot, min_overlap=0.8, min_visible=0.8, min_height=150, min_width=50):
        json_name = gt_annot['json_name']
        gt_boxes_vis = gt_annot['gt_boxes_vis']

        annotation = Annotation()
        annotation.fromJsonFile(json_name)
        size = (annotation.imgWidth, annotation.imgHeight)
        background = name2label['unlabeled'].id

        inst_img = Image.new("L", size, background)
        drawer_whole = ImageDraw.Draw(inst_img)
        cnt = 1

        instances = []

        for obj in annotation.objects:

            label = obj.label
            polygon = obj.polygon
            # polygon = np.array(polygon, dtype=np.int32)

            if label not in ['person']:
                continue

            x1, y1 = np.array(polygon).min(axis=0)
            x2, y2 = np.array(polygon).max(axis=0)

            # def PolyArea(x, y):
            #     return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            # polygon_np = np.array(polygon)
            # area = PolyArea(polygon_np[:, 0], polygon_np[:, 1])

            if obj.deleted:
                continue
            if name2label[label].id < 0:
                continue

            # id = name2label[label].id # 25, 26 for person and rider
            cnt += 1
            drawer_whole.polygon(polygon, fill=cnt)
            I = np.asarray(inst_img)
            area = (I == cnt).sum()
            # print (area, area_)

            idx, max_ov= get_corresponding_gt_box(gt_annot['gt_boxes'], box=np.array([x1, y1, x2, y2], dtype=np.float32))
            if max_ov > min_overlap and gt_annot['gt_classes'][idx] == 1 \
                and y2 - y1 > min_height and x2 - x1 > min_width \
                and x2 < annotation.imgWidth and y2 < annotation.imgHeight:
                instances.append({'inst_id': cnt,
                                  'idx': idx,
                                  'box': np.asarray([x1, y1, x2, y2], dtype=np.int32),
                                  'label': label,
                                  'area': area,
                                  'max_ov': max_ov,
                                  'polygon': polygon,
                                  })

        im = cv2.imread(gt_annot['img_name'])
        ih, iw = im.shape[0:2]
        inst_img = np.array(inst_img)

        inst_patches = []
        for inst in instances:
            idx, box, area = inst['idx'], inst['box'], inst['area']
            gt_box = gt_annot['gt_boxes'][idx]
            inst_id = inst['inst_id']
            this_inst = inst_img.copy()
            this_inst[this_inst != inst_id] = 0
            this_inst[this_inst == inst_id] = 1
            visible_area = this_inst.sum()
            if (visible_area + 0.0) / area < min_visible:
                continue
            gt_box = gt_box.astype(np.int32)
            gt_box[gt_box < 0] = 0
            gt_box[0:4:2][gt_box[0:4:2] > iw] = iw
            gt_box[1:4:2][gt_box[1:4:2] > ih] = ih
            mask_patch = this_inst[box[1]:box[3] + 1, box[0]:box[2] + 1]
            img_patch = im[box[1]:box[3] + 1, box[0]:box[2] + 1].copy()
            img_patch[mask_patch != 1] = [0, 0, 0]
            inst_patches.append({
                'label': inst['label'],
                'box': box,
                'patch': img_patch,
                'mask': mask_patch,
                'img_name': gt_annot['img_name'],
                'json_name': gt_annot['json_name'],
                'visible_area': visible_area,
                'gt_box': gt_box,
                'polygon': inst['polygon'],
                'associate_idx': idx, # the corresponding gt_boxes and gt_classes in self.gt_annots
            })
        return inst_patches

    def get_patch_path(self, img_name, idx):
        img_name = os.path.basename(img_name)
        save_name = os.path.join(self.patch_path, img_name)
        return save_name.replace('leftImg8bit', '%d_patch' % idx), \
               save_name.replace('leftImg8bit', '%d_mask' % idx)

    def load_all_instances(self):
        """add """
        cache_file = os.path.join(self.data_root, 'cache', self.split, 'persons.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.all_instances = cPickle.load(f)
            print('loading from file: ', cache_file)
            return self.all_instances
        elif not os.path.exists(os.path.join(self.data_root, 'cache', self.split)):
            os.makedirs(os.path.join(self.data_root, 'cache', self.split))

        for i, annot in enumerate(self.gt_annots):
            if i % 100 == 0:
                print('%d of %d' % (i, len(self.gt_annots)))
            inst_patches = self.get_inst_mask(annot, min_overlap=0.8, min_visible=0.9, min_height=200, min_width=50)
            # if i == 1000: break
            for j, inst in enumerate(inst_patches):
                im = inst['patch']
                mask = inst['mask']
                mask[mask == 1] = 255
                width, height = 100, 240
                p1, p2 = self.get_patch_path(inst['img_name'], i)
                assert im.shape[0] == mask.shape[0] > 0 and im.shape[1] == mask.shape[1] > 0, \
                    '{} vs {}, box:{}, gt_box:{}'.format(im.shape, mask.shape, inst['box'], inst['gt_box'])
                if not cv2.imwrite(p1, im):
                    ValueError('{} {}'.format(im.shape, mask.shape))
                if not cv2.imwrite(p2, mask.astype(np.uint8)):
                    ValueError('{}'.format(mask.shape))
                # try:
                #     im = cv2.resize(im, (width, height))
                # except:
                #     ValueError('{}'.format(im.shape))
                # I.append(im[np.newaxis, ...])
                self.all_instances.extend([inst])

        with open(cache_file, 'w') as f:
            cPickle.dump(self.all_instances, f)

        return self.all_instances

    def create_all_examples(self, repeat=3):

        extend_annot_file = os.path.join(self.extend_dir, 'extend_annots.pkl')
        if os.path.exists(extend_annot_file):
            with open(extend_annot_file, 'r') as f:
                self.extend_annots = cPickle.load(f)
                if not cfg.use_extend and self._is_training:
                    self.filter_extend_annots()
                    print ('use only original annots, contains {} images'.format(len(self.extend_annots)))
            print('loading from file: ', extend_annot_file)
            return

        for n, annot in enumerate(self.gt_annots):
            # if 'dusseldorf_000144_000019' not in annot['img_name']: continue
            # if 'aachen_000114_000019' not in annot['img_name']: continue
            # if 'aachen_000024_000019' not in annot['img_name']: continue
            # if 'cologne_000104_000019' not in annot['img_name']: continue
            for i in range(repeat):
                self.creat_an_example(annot, str(i))
                if i % 100 == 0: print (time.strftime("%H:%M:%S ") +
                                        '{} of {}: {} were created, current: {}'.format(n, len(self.gt_annots), repeat * n, annot['img_name']))

        with open(extend_annot_file, 'w') as f:
            cPickle.dump(self.extend_annots, f)

    def filter_extend_annots(self, keep_patten='extend0'):
        self.extend_annots = [a for a in self.extend_annots if keep_patten in a['img_name']]

    def get_polygons_by_boxes(self, annot):
        json_name = annot['json_name']
        gt_boxes_vis = annot['gt_boxes_vis']
        annotation = Annotation()
        annotation.fromJsonFile(json_name)
        size = (annotation.imgWidth, annotation.imgHeight)
        background = name2label['unlabeled'].id
        boxes_from_polygon = []
        all_polygons = []
        instance_ids = []
        cnt = 0
        for i, obj in enumerate(annotation.objects):
            label = obj.label
            polygon = obj.polygon

            if label not in ['person', 'rider']: continue
            if obj.deleted: continue
            if name2label[label].id < 0: continue

            cnt += 1

            x1, y1 = np.array(polygon).min(axis=0)
            x2, y2 = np.array(polygon).max(axis=0)

            box = np.asarray([x1, y1, x2, y2], dtype=np.float32)
            boxes_from_polygon.append(box)
            instance_ids.append(cnt)
            all_polygons.append(polygon)

        overlaps = cython_bbox.bbox_overlaps(
            np.ascontiguousarray(gt_boxes_vis.reshape(-1, 4), dtype=np.float),
            np.ascontiguousarray(gt_boxes_vis.reshape(-1, 4), dtype=np.float))

        n = gt_boxes_vis.shape[0]
        overlaps[np.arange(n), np.arange(n)] = 0.0

        polygons = []
        for i, gt_box_vis in enumerate(gt_boxes_vis):
            ovs = overlaps[i, :]
            idx, max_ov = get_corresponding_gt_box(boxes_from_polygon, gt_box_vis.astype(np.float32))
            # Note that: there are some unlabelled or mislabelled person masks that might crash this judgement.
            if max_ov >= 0.8 or max_ov > 0.7 and ovs.max() < 0.5 or max_ov > 0.5 and ovs.max() < 0.3:
                polygons.append(all_polygons[idx])
            else:
                polygons.append(None)
        return polygons

    def get_occlusin_labels_by_masksv2(self, annot):
        #  blockee_overlap_threshold=0.2, blocker_overlap_threshold=0.01
        """for each gt_boxes,
                1. find the associate instance in annotation
                2. for all instances
                       compute whether is blocked by this instance
                3.
        """
        json_name = annot['json_name']
        gt_boxes_vis = annot['gt_boxes_vis']
        gt_classes = annot['gt_classes']

        annotation = Annotation()
        annotation.fromJsonFile(json_name)

        size = (annotation.imgWidth, annotation.imgHeight)
        background = name2label['unlabeled'].id

        inst_img = Image.new("L", size, background)
        drawer = ImageDraw.Draw(inst_img)
        cnt = 0

        boxes_from_polygon = []
        areas_from_polygon = []
        instance_ids = []
        labels_from_polygon = []
        for i, obj in enumerate(annotation.objects):
            label = obj.label
            polygon = obj.polygon

            # if label not in ['person']:
            #     continue
            if (not label in name2label) and label.endswith('group'):
                label = label[:-len('group')]
            if obj.deleted:
                continue
            if name2label[label].id < 0:
                continue

            # if label in ['road', 'sidewalk', 'sky']: continue

            cnt += 1

            x1, y1 = np.array(polygon).min(axis=0)
            x2, y2 = np.array(polygon).max(axis=0)

            drawer.polygon(polygon, fill=cnt)
            I = np.asarray(inst_img)
            area = (I == cnt).sum()

            box = np.asarray([x1, y1, x2, y2], dtype=np.float32)
            boxes_from_polygon.append(box)
            areas_from_polygon.append(area)
            instance_ids.append(cnt)
            labels_from_polygon.append(label)

        # occlusion_labels = np.zeros([gt_boxes_vis.shape[0]], dtype=np.int32) - 1
        blocked_areas = np.zeros([gt_boxes_vis.shape[0]], dtype=np.float32)
        boxes_from_polygon = np.asarray(boxes_from_polygon)
        instance_ids = np.asarray(instance_ids)

        overlaps = cython_bbox.bbox_overlaps(
            np.ascontiguousarray(gt_boxes_vis.reshape(-1, 4), dtype=np.float),
            np.ascontiguousarray(gt_boxes_vis.reshape(-1, 4), dtype=np.float))

        n = gt_boxes_vis.shape[0]
        overlaps[np.arange(n), np.arange(n)] = 0.0

        I = np.asarray(inst_img)
        for i, gt_box_vis in enumerate(gt_boxes_vis):
            ovs = overlaps[i, :]
            cls = gt_classes[i]
            idx, max_ov= get_corresponding_gt_box(boxes_from_polygon, gt_box_vis.astype(np.float32))
            if cls <= 0 or cls >= 4: continue
            if max_ov < 0.55 and ovs.max() > 0.1:
                # which means there may be not instance segmentation for this object.
                blocked_areas[i] = 0.1
                continue
            # elif max_ov <= 0.6:
            #     blocked_areas[i] = 1 - max_ov
            #     continue

            area = areas_from_polygon[idx]
            id = instance_ids[idx]
            area_vis = (I == id).sum()
            blocked  = 1 - float(area_vis) / float(area + 0.1)
            blocked_areas[i] = blocked

        return blocked_areas

    def get_occlusion_labels_by_visible_annotation(self, annot):
        """"""
        gt_boxes = annot['gt_boxes']
        gt_boxes_vis = annot['gt_boxes_vis']
        assert gt_boxes_vis.shape[0] == gt_boxes.shape[0]
        blocked_areas = np.zeros([gt_boxes.shape[0]], dtype=np.float32)
        for i, (box, box_vis) in enumerate(zip(gt_boxes, gt_boxes_vis)):
            ov = get_overlapv2(box, box_vis)
            blocked_areas[i] = 1.0 - ov

        return blocked_areas

    # min_blocked_area = 0.3, max_blocked_area = 0.6,
    def get_isolation_labels_by_ground_plane(self, gt_boxes, gt_classes,
                                             min_blocked_area=0.0, max_blocked_area=0.2, person_class=1):
        """Using plane ground to estimate occlusion_labels denoting occlusion for each gt_box
        This function can not inference objects that are truncted by other unlabelled objects (e.g., cars, chairs)
        Only inference isolated objects
        """
        # intersections[i, j] = 0.9 means j was covered by i with 90% percents (of j area)

        intersections = cython_bbox.bbox_intersections(
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

        n = gt_boxes.shape[0]
        intersections[np.arange(n), np.arange(n)] = 0.0
        occl_mat = np.empty([n, n], dtype=np.int32)
        occl_mat.fill(-1)
        isolation_labels = np.empty(n, dtype=np.int32)
        isolation_labels.fill(-1)
        for i in range(n):
            if gt_classes[i] != person_class: continue
            q_box = gt_boxes[i]
            inds = np.where(intersections[:, i] > max_blocked_area)[0]
            blocking_inds = np.where(intersections[:, i] > min_blocked_area+0.01)[0]
            if blocking_inds.size == 0: # no objects intersect with this one
                isolation_labels[i] = 1
            else:
                bottoms = gt_boxes[inds, 3]
                inters = intersections[i, inds]
                # i is at front and blocks someone
                if np.all(bottoms < q_box[3] - 2):
                    isolation_labels[i] = 1  # blocker
                elif np.any(bottoms > q_box[3] + 2):
                    isolation_labels[i] = 0
                else:
                    isolation_labels[i] = -1

        return isolation_labels

    def get_blocking_rate_by_ground_plane(self, gt_boxes):

        overlaps = cython_bbox.bbox_overlaps(
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

        n = gt_boxes.shape[0]
        if n == 0:
            return np.zeros([0], dtype=np.float32)
        overlaps[np.arange(n), np.arange(n)] = 0.0
        blocking_rate = overlaps.max(axis=0)

        return blocking_rate


    def creat_an_example(self, annot, suffix=''):
        """
        1. randomly sample N instances from self.all_instances
        2. add to the image (keep the boxes and classes)
        3. get the occlusion_label
        """

        annotation = Annotation()
        annotation.fromJsonFile(annot['json_name'])
        im = cv2.imread(annot['img_name'])

        gt_boxes = annot['gt_boxes']
        gt_boxes_vis = annot['gt_boxes_vis']
        gt_classes = annot['gt_classes']

        blocked_areas_box = self.get_occlusion_labels_by_visible_annotation(annot)
        blocked_areas_mask = self.get_occlusin_labels_by_masksv2(annot)
        isolation_labels = self.get_isolation_labels_by_ground_plane(gt_boxes, gt_classes,
                                                                     max_blocked_area=self.block_threshold)

        occlusion_labels = np.zeros([blocked_areas_box.shape[0]], dtype=np.int32) - 1
        occlusion_labels[blocked_areas_mask > self.block_threshold] = 1
        occlusion_labels[blocked_areas_mask < self.block_threshold_low] = 0
        occlusion_labels[np.logical_and(blocked_areas_box > self.block_threshold,
                                        occlusion_labels == -1)] = 1
        occlusion_labels[np.logical_and(isolation_labels == 0, occlusion_labels == -1)] = 1
        blocking_rates = self.get_blocking_rate_by_ground_plane(gt_boxes)
        blocking_rates[gt_classes <= 0] = -1.

        polygons = self.get_polygons_by_boxes(annot)
        synth = np.zeros((gt_classes.shape[0], ), np.int32)
        assert len(polygons) == gt_classes.shape[0] == gt_classes.shape[0]

        new_boxes = []
        for i in range(blocked_areas_box.shape[0]):
            box = annot['gt_boxes'][i]
            cls = annot['gt_classes'][i]
            if cls != 1: continue
            if blocked_areas_box[i] > 0.1:  continue # already blocked
            if blocked_areas_mask[i] > 0.1: continue
            if box[3] - box[1] + 1.0 < 50: continue
            idx = np.random.randint(0, len(self.all_instances))
            inst = self.all_instances[idx]
            exclude_inds = np.setdiff1d(np.arange(gt_boxes.shape[0]), [i])
            target = random_block(box, inst,
                                  ref_boxes=annot['gt_boxes'],
                                  exclude_boxes=gt_boxes[exclude_inds, :],
                                  min_blocked_area=0.3)
            if target and suffix != '0':
                (new_box, new_polygon, blocked) = target
                # print (blocked)
                mask = inst['mask']
                mask[mask >= 128] = 1
                mask_new = cv2.resize(mask,
                                      dsize=(new_box[2] - new_box[0] + 1, new_box[3] - new_box[1] + 1),
                                      interpolation=cv2.INTER_NEAREST)
                patch = inst['patch']
                patch_new = cv2.resize(patch,
                                       dsize=(new_box[2] - new_box[0] + 1, new_box[3] - new_box[1] + 1),
                                       interpolation=cv2.INTER_NEAREST)

                # copy image patch (mask) to image
                im[new_box[1]:new_box[3]+1, new_box[0]:new_box[2]+1] = \
                    im[new_box[1]:new_box[3] + 1, new_box[0]:new_box[2] + 1] * (1 - mask_new)[:,:,np.newaxis] + \
                    mask_new[:, :, np.newaxis] * patch_new

                # add to annotation
                annotation.append_an_obj(label='person', polygon=new_polygon.tolist())
                gt_boxes = np.vstack((gt_boxes, new_box.reshape(1, 4)))
                gt_boxes_vis = np.vstack((gt_boxes_vis, new_box.reshape(1, 4)))
                gt_classes = np.append(gt_classes, 1)
                if blocked > self.block_threshold: occlusion_labels[i] = 1
                occlusion_labels = np.append(occlusion_labels, 0)
                inter = self.get_inter(new_box, box)
                blocking_rates = np.append(blocking_rates, inter / ((new_box[2]-new_box[0]+1.0) * (new_box[3]-new_box[1]+1.0)))
                blocking_rates[i] = inter / ((box[2]-box[0]+1.0) * (box[3]-box[1]+1.0))
                get_overlap(new_box, box)
                new_boxes.append(new_box)
                polygon = [(p[0], p[1]) for p in new_polygon]
                polygons.append(polygon)
                synth = np.append(synth, 1)


        # save to files (both image and annotations)
        extend_img_name, extend_json_name = self._get_extend_save_path(annot['img_name'], suffix)
        cv2.imwrite(extend_img_name, im)
        annotation.toJsonFile(extend_json_name)
        assert len(polygons) == gt_classes.shape[0] == gt_classes.shape[0]
        d = {'gt_classes': gt_classes,
             'gt_boxes': gt_boxes,
             'gt_boxes_vis': gt_boxes_vis,
             'occlusion_labels': occlusion_labels,
             'img_name': extend_img_name,
             'json_name': extend_json_name,
             'polygons': polygons,
             'blocking_rates': blocking_rates,
             'synth': synth
         }
        self.extend_annots.append(d)

        if not check_boxes(gt_boxes):
            ValueError('{} {}'.format(extend_img_name, gt_boxes))

        # extend_annot_name = extend_json_name.replace('json', 'pkl')
        # with open(extend_annot_name, 'w') as f:
        #     cPickle.dump(d, f)

        im_vis, im_instance = self._draw_an_example_with_occlusions(d)
        vis_img_name = self._get_vis_save_path(annot['img_name'], suffix)
        cv2.imwrite(vis_img_name, im_vis)
        vis_img_name = vis_img_name.replace('.png', '_instance.png')
        cv2.imwrite(vis_img_name, im_instance)

        # extend_annot_name = extend_annot_name.replace('.pkl', '_all.json')
        # with open(extend_annot_name, 'w') as f:
        #     for k in d.keys():
        #         if isinstance(d[k], np.ndarray):
        #             d[k] = d[k].tolist()
        #     try:
        #         json.dump(d, f, indent=1, separators=(',', ': '), default=lambda o: o.__dict__,)
        #     except:
        #         print (d)
        #         raise ValueError('')

    def get_inter(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        a1 = (b1[2] - b1[0] + 1.0) * (b1[3] - b1[1] + 1.0)
        a2 = (b2[2] - b2[0] + 1.0) * (b2[3] - b2[1] + 1.0)
        area = 0.0
        if x2 > x1 and y2 > y1:
            area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        return area

    def _draw_an_example_with_occlusions(self, extend_annot):
        im = cv2.imread(extend_annot['img_name'])
        gt_boxes = extend_annot['gt_boxes']
        gt_classes = extend_annot['gt_classes']
        occlusion_labels = extend_annot['occlusion_labels']
        blocking_rates = extend_annot['blocking_rates']
        synth = extend_annot['synth']

        h, w, _ = im.shape
        thick = 2
        for occ, cls, box, brate, fake in zip(occlusion_labels, gt_classes, gt_boxes, blocking_rates, synth):
            if cls not in [1, 2, 3]: continue
            box = box.astype(np.int32)
            if occ == 0:
                color = (72, 119, 5)
                txt = 'bloker'
            elif occ == 1:
                color = (100, 100, 0)
                txt = 'blokee'
            elif occ < 0:
                color = (128, 128, 128)
                txt = 'ignore'
            else:
                ValueError('unknow occlusion label{}'.format(occ))
            if cls == 2: txt, color = 'rider', (128, 128, 128)
            if cls == 3: txt, color = 'sit', (128, 128, 128)

            if fake == 1:
                color = (80, 75, 120)

            btxt = '%.2f' % brate

            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color=color, thickness=thick)
            # cv2.putText(im, txt, (box[0], box[1] - 8), 0, 6e-4 * h, color, thickness=thick)
            # cv2.putText(im, btxt, (box[0], box[3] - 8), 0, 6e-4 * h, color, thickness=thick)

        polygons = extend_annot['polygons']
        background = (112, 25, 25)
        labelImg = Image.new("RGBA", (w, h), background)
        drawer = ImageDraw.Draw(labelImg)
        for i, (polygon, fake) in enumerate(zip(polygons, synth)):
            if polygon is None: continue
            cls = gt_classes[i]
            occ = occlusion_labels[i]
            if cls not in [1, 2, 3]: continue
            if occ == 0:
                color = (72, 119, 5)
            elif occ == 1:
                color = (100, 100, 0)
            elif occ < 0:
                color = (128, 128, 128)

            if cls == 2: txt, color = 'rider', (128, 128, 128)
            if cls == 3: txt, color = 'sit', (128, 128, 128)

            if fake == 1:
                color = (80, 75, 120)

            drawer.polygon(polygon, fill=color)

        return im, np.asarray(labelImg)


    def _get_extend_save_path(self, img_name, suffix='0'):
        img_stem = os.path.basename(img_name).replace('.png', '')
        return os.path.join(self.extend_dir, img_stem + '_extend%s.png' % suffix), \
               os.path.join(self.extend_dir, img_stem + '_extend%s.json' % suffix)

    def _get_vis_save_path(self, img_name, suffix='0'):
        img_stem = os.path.basename(img_name).replace('.png', '')
        return os.path.join(self.vis_dir, img_stem + '_vis%s.png' % suffix)

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
                dict = {
                    "image_id": img_id,
                    "category_id": id,
                    "bbox": [round(x, 1), round(y, 1), round(width, 1), round(height, 1)],
                    "score": round(score, 3)
                }
                dict_list.append(dict)
        return dict_list

    def filter_result(self, result_file, min_height=45, max_height=78):
        anns = json.load(open(result_file))
        res = []
        for a in anns:
            height = a['bbox'][3]
            if height >= min_height and height <= max_height:
                res.append(a)
        return res

    def eval(self, result_file):
        """evaluation detection results"""

        annFile = os.path.join('data/citypersons/annotations', 'val_gt.json')
        resFile = os.path.join('output/citypersons', 'results.txt')
        if not os.path.exists('output/citypersons'):
            os.makedirs('output/citypersons')

        with open(resFile, 'w') as res_file:
            for id_setup in range(0, 4):
                cocoGt = COCO(annFile)
                cocoDt = cocoGt.loadRes(result_file)
                imgIds = sorted(cocoGt.getImgIds())
                cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
                cocoEval.params.imgIds = imgIds
                cocoEval.params.expFilter = 1.25
                cocoEval.evaluate(id_setup)
                cocoEval.accumulate()
                print ('id_setup:', id_setup)
                cocoEval.summarize(id_setup, res_file)

    def eval_over_scales(self, result_file):
        annFile = os.path.join('data/citypersons/annotations', 'val_gt.json')
        resFile = os.path.join('output/citypersons', 'results_over_scales.txt')
        if not os.path.exists('output/citypersons'):
            os.makedirs('output/citypersons')
        with open(resFile, 'w') as res_file:
            # 4: 25 - 50
            # 5: 50 - 75
            # 6: 75 - 125
            # 7: 125 - 225
            # 8: 225 - 10000
            for id_setup in range(4, 10):
                cocoGt = COCO(annFile)
                cocoDt = cocoGt.loadRes(result_file)
                imgIds = sorted(cocoGt.getImgIds())
                cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
                cocoEval.params.imgIds = imgIds
                cocoEval.params.expFilter = 1.25
                cocoEval.evaluate(id_setup)
                cocoEval.accumulate()
                print('id_setup:', id_setup)
                cocoEval.summarize(id_setup, res_file)

    def exclude_negative_images(self):
        num = len(self.extend_annots)
        extend_annots = []
        for i in range(num):
            annot = self.extend_annots[i]
            gt_boxes = annot['gt_boxes'].astype(np.float32)
            gt_boxes_vis = annot['gt_boxes_vis'].astype(np.float32)
            gt_classes = annot['gt_classes'].astype(np.int32)
            gt_classes[gt_classes == 0] = -1
            gt_classes[gt_classes == 2] = 1
            gt_classes[gt_classes == 3] = 1
            gt_classes[gt_classes == 4] = -1
            gt_classes[gt_classes == 5] = -1

            areas = ((gt_boxes_vis[:, 3] - gt_boxes_vis[:, 1] + 1) * (gt_boxes_vis[:, 2] - gt_boxes_vis[:, 0] + 1)) / \
                    ((gt_boxes[:, 3] - gt_boxes[:, 1] + 1) * (gt_boxes[:, 2] - gt_boxes[:, 0] + 1))
            heights = (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

            gt_classes[areas < 0.3] = -1
            gt_classes[heights < 32] = -1
            valid_num = (gt_classes > 0).astype(np.int32).sum()
            if valid_num > 0:
                extend_annots.append(annot)
        self.extend_annots = extend_annots
        print('exclude negative images %d --> %d' % (num, len(extend_annots)))

    def __len__(self):
        if self.split == 'val' and not self._is_training:
            return len(self.gt_annots)
        return len(self.extend_annots)

    def __getitem__(self, i):
        """return image, gt_classes, gt_boxes, occlusion_labels"""

        if self.split == 'val' and not self._is_training:
            return self._getitem_for_val_(i)
        """
        d = {'gt_classes': gt_classes,
             'gt_boxes': gt_boxes,
             'gt_boxes_vis': gt_boxes_vis,
             'occlusion_labels': occlusion_labels,
             'img_name': extend_img_name,
             'json_name': extend_json_name,
             'polygons': polygons,
         }
         """
        annot = self.extend_annots[i]
        gt_boxes = annot['gt_boxes'].astype(np.float32)
        gt_boxes_vis = annot['gt_boxes_vis'].astype(np.float32)
        gt_classes = annot['gt_classes'].astype(np.int32)

        img_name = annot['img_name']
        gt_classes[gt_classes == 0] = -1
        gt_classes[gt_classes == 2] = 1
        gt_classes[gt_classes == 3] = 1
        gt_classes[gt_classes == 4] = -1
        gt_classes[gt_classes == 5] = -1

        areas = ((gt_boxes_vis[:, 3] - gt_boxes_vis[:, 1] + 1) * (gt_boxes_vis[:, 2] - gt_boxes_vis[:, 0] + 1)) / \
                ((gt_boxes[:, 3] - gt_boxes[:, 1] + 1) * (gt_boxes[:, 2] - gt_boxes[:, 0] + 1))
        heights = (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

        gt_classes[areas < 0.3] = -1
        gt_classes[heights < 32] = -1

        assert os.path.exists(img_name)

        n = gt_classes.shape[0]
        height, width = cv2.imread(img_name).shape[:2]
        inst_masks = np.zeros([n, height, width], dtype=np.int32)
        mask = np.zeros([height, width], dtype=np.int32)

        im, im_scale, annots = data_layer_keep_aspect_ratio(img_name, gt_boxes, gt_classes, inst_masks, mask,
                                                            self._is_training)

        # im = np.transpose(im, [2, 0, 1])  # c, h, w
        img_id = os.path.splitext(os.path.basename(img_name))[0]

        return im, im_scale, annots, img_id

    def _getitem_for_val_(self, i):
        """return image only"""
        annot = self.gt_annots[i]
        img_name = annot['img_name']
        im = cv2.imread(img_name)
        gt_boxes = annot['gt_boxes']
        gt_classes = annot['gt_classes']
        gt_boxes_vis = annot['gt_boxes_vis'].astype(np.float32)
        areas = ((gt_boxes_vis[:, 3] - gt_boxes_vis[:, 1] + 1) * (gt_boxes_vis[:, 2] - gt_boxes_vis[:, 0] + 1)) / \
                ((gt_boxes[:, 3] - gt_boxes[:, 1] + 1) * (gt_boxes[:, 2] - gt_boxes[:, 0] + 1))
        heights = (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

        gt_classes[gt_classes == 0] = -1
        gt_classes[gt_classes == 2] = 1
        gt_classes[gt_classes == 3] = 1
        gt_classes[gt_classes == 4] = 1
        gt_classes[gt_classes == 5] = -1
        # gt_classes[areas < 0.3] = -1
        # gt_classes[heights < 20] = -1
        # gt_classes[areas < 0.35] = -1
        # gt_classes[heights < 50] = -1
        # gt_classes[heights > 75] = -1

        n = gt_classes.shape[0]
        height, width = cv2.imread(img_name).shape[:2]
        inst_masks = np.zeros([n, height, width], dtype=np.int32)
        mask = np.zeros([height, width], dtype=np.int32)
        im, im_scale, annots = data_layer_keep_aspect_ratio(img_name, gt_boxes, gt_classes, inst_masks, mask,
                                                            is_training=False)

        # im = np.transpose(im, [2, 0, 1])  # c, h, w

        # img_id = os.path.splitext(os.path.basename(img_name))[0]
        img_id = i + 1

        return im, im_scale, annots, img_id

    def build_occlusion_mask(self, gt_boxes, gt_classes, iw, ih):
        n = gt_boxes.shape[0]
        mask = np.zeros([ih, iw], dtype=np.int32)

        for i in range(n):
            box = gt_boxes[i].astype(np.int32)
            cls = gt_classes[i]
            if cls <= 0: continue
            x1, y1 = max(box[0], 0), max(box[1], 0)
            x2, y2 = min(box[2], iw), min(box[3], ih)
            mask[y1:y2, x1:x2] = 1

        for i in range(n):
            box = gt_boxes[i].astype(np.int32)
            cls = gt_classes[i]
            if cls <= 0: continue
            x1, y1 = max(box[0], 0), max(box[1], 0)
            x2, y2 = min(box[2], iw), min(box[3], ih)
            # mask[y1:y2, x1:x2] = 1

            for j in range(i+1, n):
                box2 = gt_boxes[j].astype(np.int32)
                cls2 = gt_classes[j]
                if cls2 <= 0: continue
                x1, y1 = max(box2[0], 0), max(box2[1], 0)
                x2, y2 = min(box2[2], iw), min(box2[3], ih)
                # mask[y1:y2, x1:x2] = 1

                x1, y1 = max(box[0], box2[0]), max(box[1], box2[1])
                x2, y2 = min(box[2], box2[2]), min(box[3], box2[3])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, iw), min(y2, ih)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 2

        for i in range(n):
            box = gt_boxes[i].astype(np.int32)
            cls = gt_classes[i]
            if cls > 0: continue
            x1, y1 = max(box[0], 0), max(box[1], 0)
            x2, y2 = min(box[2], iw), min(box[3], ih)
            mask[y1:y2, x1:x2] = -1

        return mask

    def build_head_point_map(self, gt_boxes, gt_classes, iw, ih):

        sigma = 10.

        n = gt_boxes.shape[0]
        conf_map = np.zeros([ih, iw], dtype=np.float32)

        for i in range(n):
            box = gt_boxes[i].astype(np.int32)
            cls = gt_classes[i]
            if cls <= 0: continue
            x = int(0.5 * (box[0] + box[2]))
            y = int(max(box[1], 0))
            xs = np.arange(iw)
            ys = np.arange(ih)
            xs, ys = np.meshgrid(xs, ys)
            xs -= x
            ys -= y
            D2 = xs ** 2. + ys ** 2.
            vs = np.exp(-D2 / sigma / sigma)

            conf_map += vs

        return conf_map

    def build_p_gt_boxes(self, gt_boxes, gt_classes):

        inds = np.where(gt_classes > 0)[0]
        p_gt_boxes = []
        modes = []
        p_gt_classes = []
        if inds.size > 0:
            gt_boxes = gt_boxes[inds]
            gt_classes = gt_classes[inds]
            overlaps = cython_bbox.bbox_overlaps(
                np.ascontiguousarray(gt_boxes[:, :4].reshape(-1, 4), dtype=np.float),
                np.ascontiguousarray(gt_boxes[:, :4].reshape(-1, 4), dtype=np.float))
            n = inds.size
            overlaps[np.arange(n), np.arange(n)] = 0.0
            visited = np.zeros((n, ), dtype=np.int32)

            for i in range(n):
                box1 = gt_boxes[i]
                if visited[i] == 1: continue
                visited[i] = 1
                for j in range(i+1, n):
                    box2 = gt_boxes[j]
                    if overlaps[i, j] < cfg.block_threshold_lo:
                        continue
                    else:
                        # if box1[3] > box2[3]:
                        #     p_box = np.concatenate([box1, box2], axis=0)
                        # else:
                        #     p_box = np.concatenate([box2, box1], axis=0)

                        p_box = np.concatenate([box1, box2], axis=0)
                        p_gt_boxes.append(p_box)
                        cls = 1
                        mode = 1
                        if overlaps[i, j] < cfg.block_threshold_hi:
                            cls = -1
                            mode = 0
                        modes.append(mode)
                        p_gt_classes.append(cls)

            p_gt_boxes = np.asarray(p_gt_boxes, dtype=np.float32).reshape(-1, 8)
            p_gt_classes = np.asarray(p_gt_classes, dtype=np.int32)
            modes = np.asarray(modes, dtype=np.int32)
            return p_gt_boxes, p_gt_classes, modes

        return np.zeros([0, 8], dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)


def random_block(box, inst, ref_boxes, exclude_boxes,
                 min_blocked_area=0.3, max_blocked_area=0.6,
                 num_attempts=200, ih=1024, iw=2048):
    h = box[3] - box[1] + 1.0
    w = box[2] - box[0] + 1.0
    cx = (box[0] + 0.5 * w)
    cy = (box[1] + 0.5 * h)
    polygon = inst['polygon']
    x1, y1 = np.array(polygon).min(axis=0)
    x2, y2 = np.array(polygon).max(axis=0)

    # scale and shift polygon to original
    scale = h / (y2 - y1 + 1.0)
    scale = (np.random.rand() - 0.5) * 0.05 * scale + scale
    polygon_new = np.array(polygon).copy()
    polygon_new[:, 0] = polygon_new[:, 0] - x1
    polygon_new[:, 1] = polygon_new[:, 1] - y1
    polygon_new = (polygon_new * scale).astype(np.int32)
    x1, y1 = polygon_new.min(axis=0)
    x2, y2 = polygon_new.max(axis=0)
    assert x1 < x2 and y1 < y2, '{}'.format([x1, y1, x2, y2])

    shift_x = np.random.uniform(-1.0 * w, 1.0 * w, (num_attempts,))
    shift_y = np.random.uniform(-0.01 * h, 0.1 * h, (num_attempts,))
    boxes = np.zeros((num_attempts, 4), np.float32)
    boxes[:, 0] = shift_x + box[0]
    boxes[:, 1] = shift_y + box[1]
    boxes[:, 2] = boxes[:, 0] + x2 - x1
    boxes[:, 3] = boxes[:, 1] + y2 - y1

    boxes = clip_and_filter(boxes, ih, iw).reshape(-1, 4)
    overlaps = cython_bbox.bbox_intersections(
        np.ascontiguousarray(boxes, dtype=np.float),
        np.ascontiguousarray(ref_boxes.reshape(-1, 4), dtype=np.float))
    # bb_ovs = overlaps[:, 0]
    bb_ovs = overlaps.max(axis=1)
    inds = np.where(np.logical_and(bb_ovs > min_blocked_area,
                                   bb_ovs < max_blocked_area))[0]
    if inds.size > 0:
        id = np.random.choice(inds)
        chozen_box = boxes[id, :].astype(np.int32)
        polygon_new[:, 0] = polygon_new[:, 0] + chozen_box[0]
        polygon_new[:, 1] = polygon_new[:, 1] + chozen_box[1]

        if exclude_boxes.size > 0:
            overlaps = cython_bbox.bbox_overlaps(
                np.ascontiguousarray(exclude_boxes.reshape(-1, 4), dtype=np.float),
                np.ascontiguousarray(chozen_box.reshape(-1, 4), dtype=np.float))
            ovs = overlaps.max(axis=1)
            if ovs.max() > 0.5:
                return None

            intersections = cython_bbox.bbox_intersections(
                np.ascontiguousarray(exclude_boxes.reshape(-1, 4), dtype=np.float),
                np.ascontiguousarray(chozen_box.reshape(-1, 4), dtype=np.float))
            intersections

        return chozen_box, polygon_new, get_blocked(box, boxes[id, :])

    return None

def collate_fn(data):

    # im, TARGETS, ori_im, ANNOTATIONS, img_id
    im_batch, im_scale_batch, anchors, rpn_targets, inst_masks_batch, mask_batch = \
        data_layer_keep_aspect_ratio_batch(data, is_training=True)

    input = torch.stack(everything2tensor(im_batch))

    # [x1, y1, x2, y2, cls]
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

    # image ids
    inst_masks, mask, downsampled_mask = [], [], []

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
    cpe = citypersons_extend(data_dir, split, is_training)
    print('loading all instances...')
    cpe.load_all_instances()
    print('loading all instances... done')
    print('create images...')
    cpe.create_all_examples(repeat=3)
    print('create images... done')
    cpe.exclude_negative_images()
    if is_training:
        return sDataLoader(cpe, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)
    else:
        return sDataLoader(cpe, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn_testing)

def clip_and_filter(boxes, ih, iw):
    # boxes[:, 1][boxes[:, 1] < 0] = 0
    # boxes[:, 0][boxes[:, 0] < 0] = 0
    # boxes[:, 3][boxes[:, 3] > ih] = ih - 1
    # boxes[:, 2][boxes[:, 2] > iw] = iw - 1
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    inds = np.where(
        np.logical_and(
            np.logical_and(
                np.logical_and(ws > 0, hs > 0),
                np.logical_and(x1 > 0, y1 > 0),
            ),
            np.logical_and(x2 < iw, y2 < ih),
        )
    )[0]
    return boxes[inds, :]


def add_brightness(im):
    # distort brightness
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    v_[v_ > 0] += 20
    v_[v_ > 255] = 255
    v_[v_ < 0] = 0
    hsv = cv2.merge((h_, s_, v_))
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return im


def get_corresponding_gt_box(gt_boxes, box):
    max_ov = 0.0
    second_max_ov = 0.0
    idx = -1
    area_box = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
    for i, gt_box in enumerate(gt_boxes):
        x1, y1 = max(gt_box[0], box[0]), max(gt_box[1], box[1])
        x2, y2 = min(gt_box[2], box[2]), min(gt_box[3], box[3])

        area_gt_box = (gt_box[2] - gt_box[0] + 1.0) * (gt_box[3] - gt_box[1] + 1.0)

        area = 0.0
        if x2 > x1 and y2 > y1:
            area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
            ov = (area + 0.0) / (area_gt_box + area_box - area)
            if ov >= max_ov:
                second_max_ov = max_ov
                max_ov = ov
                idx = i

    return idx, max_ov


def get_overlap(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    a1 = (b1[2] - b1[0] + 1.0) * (b1[3] - b1[1] + 1.0)
    a2 = (b2[2] - b2[0] + 1.0) * (b2[3] - b2[1] + 1.0)
    area = 0.0
    if x2 > x1 and y2 > y1:
        area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    return area / (a1 + a2 - area)

def get_overlapv2(b1, b2):
    ov = get_overlap(b1, b2)
    hov = (b2[3] - b2[1] + 1.0) / (b1[3] - b1[1] + 1.0)
    return hov


def get_blocked(box, query_box):
    x1, y1 = max(box[0], query_box[0]), max(box[1], query_box[1])
    x2, y2 = min(box[2], query_box[2]), min(box[3], query_box[3])
    a1 = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
    area = 0.0
    if x2 > x1 and y2 > y1:
        area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    return area / a1

def simple_matting(im, patch, mask, box):
    h, w, _ = im.shape[0:]
    box_ = box.copy()
    bh = box[3] - box[1] + 1
    bw = box[2] - box[0] + 1
    box_[0] = max(box_[0] - bw * 0.5, 0)
    box_[2] = min(box_[2] + bw * 0.5, w-1)
    box_[1] = max(box_[1] - bh * 0.5, 0)
    box_[3] = min(box_[3] + bh * 0.5, h - 1)

def check_boxes(boxes):
    if boxes.size == 0:
        return True
    heights = boxes[:, 3] - boxes[:, 1] + 1
    widths = boxes[:, 2] - boxes[:, 0] + 1
    if widths.min() <= 0 or heights.min() <= 0:
        return False
    return True


def statistics():
    import sys
    from libs.layers.data_layer import data_layer
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    cp = citypersons_extend('./data/citypersons/', 'train', data_layer, True)
    cp.load_all_instances()
    cp.create_all_examples(repeat=2)
    Heavy_num = 0
    Occl_num = 0
    Heavy_num_ori = 0
    Occl_num_ori = 0
    All_num = 0
    All_num_ori = 0
    for i, d in enumerate(cp):
        im, TARGETS, ori_im, ANNOTATIONS, img_id = d
        [bboxes, classes] = ANNOTATIONS

        heights = bboxes[:, 3] - bboxes[:, 1] + 1
        keep = np.logical_and(classes > 0, heights > 50)
        inds = np.where(np.logical_and(classes > 0, heights > 50))[0]
        bboxes = bboxes[inds, :]
        classes = classes[inds]

        n = bboxes.shape[0]
        assert classes.size == n
        if n == 0: continue
        overlaps = cython_bbox.bbox_overlaps(
            np.ascontiguousarray(bboxes.reshape(-1, 4), dtype=np.float),
            np.ascontiguousarray(bboxes.reshape(-1, 4), dtype=np.float))

        overlaps[np.arange(n), np.arange(n)] = 0.0
        max_overlaps = overlaps.max(axis=1)
        heavy_num = max_overlaps[max_overlaps > 0.35].size
        occl_num = max_overlaps[max_overlaps > 0.2].size
        Heavy_num += heavy_num
        Occl_num += occl_num
        All_num += n
        if i % 2 == 0:
            Heavy_num_ori += heavy_num
            Occl_num_ori += occl_num
            All_num_ori += n
        if i % 100 == 0:
            print (i, img_id, 'heavy:', Heavy_num, 'occl:', Occl_num, 'All:', All_num,
                   'heavy_ori:', Heavy_num_ori, 'occl_ori:', Occl_num_ori, 'All_ori:', All_num_ori)

    # print(Heavy_num)
    # print(Occl_num)
    print(Heavy_num, Occl_num)
    print(Heavy_num_ori, Occl_num_ori)
    print (All_num, All_num_ori)


if __name__ == '__main__':

    statistics()
    # import sys
    # from libs.layers.data_layer import data_layer
    # sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    # # d = citypersons('./data/citypersons/', split='train')
    # splits = ['train', 'val']
    # for split in splits:
    #     cp = citypersons_extend('./data/citypersons/', split, data_layer, True)
    #     print('loading all instances...')
    #     cp.load_all_instances()
    #     print('loading all instances... done')
    #     print('create images...')
    #     cp.create_all_examples(repeat=2)
    #     print('create images... done')
    #
    #     num_bg, num_fg1, num_fg2 = 0, 0, 0
    #     for i, d in enumerate(cp):
    #         m = d[4][0]
    #         num_bg += (m == 0).long().sum()
    #         num_fg1 += (m == 1).long().sum()
    #         num_fg2 += (m == 2).long().sum()
    #         if i % 20 == 0 or i == len(cp):
    #             print ('%d of %d' % (i, len(cp)))
    #             print (num_bg, num_fg1, num_fg2)


    from IPython import embed; embed()