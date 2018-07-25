import cv2
import numpy as np
from libs.boxes.bbox_transform import clip_boxes

import libs.layers.anchor as anchor
import libs.configs.config as cfg
# from libs.boxes.cython_nms import nms, nms_new, nms_multianchor
from libs.boxes.cython_nms import nms_multianchor, nms_new
from libs.boxes.nms_wrapper import nms
import libs.visualization.vis as vis
from libs.preprocessings.fixed_size import image_transform_inv

def compute_detection(Outputs, Anchors, ih, iw, num_classes,
                      score_threshold=0.5, overlap_threshold=0.7, only_anchor=False, max_det_num=100):
    """Post processing outputs of single shot detection
    Params:
    Outpus: a list of (rpn_cls, rpn_boxes)
    Outpus: a list of anchors
    """
    _, rpn_box, rpn_prob = Outputs
    n = cfg.batch_size

    assert Anchors.shape[0] * n == rpn_prob.shape[0] == rpn_box.shape[0], \
        'shape dont match'

    rpn_box_ = rpn_box.copy()
    if only_anchor:
        rpn_box_[...] = 0.0

    out_dim = rpn_box.shape[0] / n
    Dets = []
    for i in range(n):
        s, e = i * out_dim, (i + 1) * out_dim
        bboxes = rpn_box_[s:e]
        probs = rpn_prob[s:e]
        final_boxes, _, _ = anchor.decode(bboxes, probs, Anchors, ih, iw, num_classes)
        dets = []
        for j in xrange(1, num_classes):
            inds = np.where(probs[:, j] > score_threshold)[0]
            probs_selected = probs[inds, j]
            boxes_selected = final_boxes[inds, :]
            cls_dets = np.hstack((boxes_selected, probs_selected[:, np.newaxis])).astype(np.float32, copy=False)
            if False:
                keep, clusters = nms_multianchor(cls_dets, overlap_threshold)
                if only_anchor:
                    cls_dets = cls_dets[keep, :]
                else:
                    cls_dets = dets_cluster(cls_dets, keep, clusters)
            if True:
                keep = nms(cls_dets, overlap_threshold)
                cls_dets = cls_dets[keep, :]
            tmp_det = np.zeros([cls_dets.shape[0], 6])
            tmp_det[:, 0:5] = cls_dets
            tmp_det[:, 5] = j  # class-label
            dets.append(tmp_det)
        dets = np.vstack(dets)
        order = dets[:, 4].argsort()[::-1]
        dets = dets[order, :]
        Dets.append(dets[:max_det_num, :])
    return Dets

def compute_detection_new(batch_rpn_box, batch_rpn_prob, batch_anchors, image_inds,
                          ih, iw, num_classes,
                          score_threshold=0.5, overlap_threshold=0.7, only_anchor=False,
                          max_dets=100):
    """Post processing outputs of single shot detection, this decode boxes only once
    """
    assert image_inds.shape[0] == batch_rpn_box.shape[0] == batch_rpn_prob.shape[0] == batch_anchors.shape[0], \
        'unmatch output: {} vs {} vs {} vs {}'.format(image_inds.shape[0], batch_rpn_box.shape[0],
                                                      batch_rpn_prob.shape[0], batch_anchors.shape[0])

    n = cfg.batch_size
    rpn_box_ = batch_rpn_box.copy()
    if only_anchor:
        rpn_box_[...] = 0.0

    Dets = []
    for i in range(n):
        inds = np.where(image_inds == i)[0]
        bboxes = rpn_box_[inds]
        probs = batch_rpn_prob[inds]
        anc = batch_anchors[inds]

        probs_positive = 1 - probs[:, 0] # the objectness score = 1 - background
        final_boxes, _, _ = anchor.decode(bboxes, probs, anc, ih, iw, num_classes)

        object_dets = np.hstack((final_boxes, probs_positive[:, np.newaxis])).astype(np.float32, copy=False)
        if False:
            keep, clusters = nms_multianchor(object_dets, overlap_threshold)
            keep = np.asarray(keep, dtype=np.int32)
            if only_anchor:
                object_dets = object_dets[keep, :]
            else:
                object_dets = dets_cluster(object_dets, keep, clusters)
        if True:
            keep = nms(object_dets, overlap_threshold)
            keep = np.asarray(keep, dtype=np.int32)
            object_dets = object_dets[keep, :]

        if keep.size > max_dets:
            keep = keep[:max_dets]

        probs = probs[keep]
        probs_objectness = probs[:, 1:]

        # if True:
        #     cls_dets = np.zeros([object_dets.shape[0], 6])
        #     cls_dets[:, 0:4] = object_dets[:, 0:4]
        #     cls_argmax = probs_objectness.argmax(axis=1)
        #     cls_maxprobs = probs_objectness[np.arange(cls_argmax.shape[0]), cls_argmax]
        #
        # else:
        d0, d1 = np.where(probs_objectness > score_threshold)
        cls_maxprobs = probs_objectness[d0, d1]
        cls_argmax = d1
        cls_dets = np.zeros([d0.size, 6])
        cls_dets[:, 0:4] = object_dets[d0, 0:4]

        cls_dets[:, 4] = cls_maxprobs
        cls_dets[:, 5] = cls_argmax + 1

        order = cls_dets[:, 4].argsort()[::-1]
        cls_dets = cls_dets[order, :]

        Dets.append(cls_dets)
    return Dets

def nms_advanced(Dets):
    # intersection
    Dets_new = []
    for dets in Dets:
        cls_ids = dets[:, 5].astype(np.int32)
        ids = np.unique(cls_ids)
        dets_new = []
        for id in ids:
            overlap_threshold = cfg.overlap_threshold_noncrowd if id in cfg.noncrowd_classes else cfg.overlap_threshold
            inds = np.where(cls_ids == id)[0]
            dt = dets[inds]
            if inds.size > 1:
                try:
                    keep, _ = nms_multianchor(dt[:, 0:5].astype(np.float32), overlap_threshold)
                except:
                    print (dt)
                    raise
                # keep = nms(dt[:, 0:5], overlap_threshold)
                dt = dt[keep]
            dets_new.append(dt)
        if len(dets_new) > 0:
            Dets_new.append(np.vstack(dets_new))
        else:
            Dets_new.append(np.zeros((0, 6), dtype=dets.dtype))
    return Dets_new

def nms_stitch(Dets):
    """this function hurt the accuracy..."""
    thrsh_scores = [(0.1, 0.5, 0.95)]
    Dets_new = []
    for dets in Dets:
        probs = dets[:, 4].astype(np.float32)
        all_inds = np.arange(probs.shape[0])
        drops_inds = []
        for (score1, score2, th) in thrsh_scores:
            if score2 < cfg.score_threshold:
                continue
            th = max(cfg.overlap_threshold, th)
            inds = np.where(np.logical_and(probs > score1, probs <= score2))[0]
            if inds.size > 0 and probs.shape[0] > 1:
                keep_inds, _ = nms_multianchor(dets[:, 0:5].astype(np.float32), th)
                ignored_inds = np.setdiff1d(inds, keep_inds)
                drops_inds.append(ignored_inds)

        if len(drops_inds) > 0:
            drops_inds = np.unique(np.hstack(drops_inds))
            keep_inds = np.setdiff1d(all_inds, drops_inds)
            dets = dets[keep_inds]
        Dets_new.append(dets)
    return Dets_new

def thresholding(Dets):
    #
    Dets_new = []
    for dets in Dets:
        cls_ids = dets[:, 5].astype(np.int32)
        ids = np.unique(cls_ids)
        dets_new = []
        for id in ids:
            threshold = cfg.score_threshold_classes[id] - 0.1
            inds = np.where(cls_ids == id)[0]
            dt = dets[inds]
            keep = np.where(dt[:, 4] >= threshold)[0]
            dt = dt[keep]
            dets_new.append(dt)
        if len(dets_new) > 0:
            Dets_new.append(np.vstack(dets_new))
        else:
            Dets_new.append(np.zeros((0, 6), dtype=dets.dtype))
    return Dets_new


def segmentation(Masks, num_classes):
    """Masks: a list of (N, C, H, W)"""
    M = Masks[0]
    n = M.shape[0]
    Segs = [None for i in range(n)]
    for i in range(n):
        m = M[i, :, :, :]
        m = m.argmax(axis=0)
        Segs[i] = m
    return np.asarray(Segs)

def draw_detection(inputs, Dets, class_names=None):
    """
    :param inputs: of shape (N, C, H, W) ndarray, the channel is in rgb order
    :param Dets: is a N x C 2-d list, each element is Mx5 detection result [x1, y1, x2, y2, cls_id]
    :return: 
    """
    # draw image
    n = len(Dets)
    assert inputs.ndim == 4 and inputs.shape[0] == n, \
        'not match {} vs {}, {} vs {}'.format(inputs.ndim, 4, inputs.shape[0], n)
    Is = []
    # print ('detection:', [d.shape[0] for d in Dets])

    for i, dets in enumerate(Dets):
        img = inputs[i, :, :, :].copy()
        img = np.transpose(img, [1, 2, 0])
        # img = (img + 1.0) * 128.0
        # img = img.astype(np.uint8)
        img = image_transform_inv(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.ascontiguousarray(img)
        h, w, _ = img.shape
        img = vis.draw_detection(img, dets[:, 0:4].astype(np.int), cls_inds=dets[:, 5].astype(np.int),
                                 scores=dets[:, 4], cls_name=class_names)
        Is.append(img)
    return np.asarray(Is)


def draw_masks(Segs, class_names=None):
    """NxHxW ndarray"""
    # set to cls-1 to 101
    n, h, w = Segs.shape
    Is = []
    num_classes = len(class_names) if class_names is not None else -1
    for i in range(n):
        Segs_color = np.zeros((h, w, 3), dtype=np.int)
        for y in range(h):
            for x in range(w):
                cls = Segs[i, y, x]
                Segs_color[y, x, :] = vis.get_color(cls, num_classes)

        Segs_color = cv2.resize(Segs_color, (4*w, 4*h), interpolation=cv2.INTER_NEAREST)
        Is.append(Segs_color)
    return  np.ascontiguousarray(Is)

def draw_gtboxes(inputs, gt_boxes, data_format='NCHW', class_names=None):
    """
    Draw boxes on images
    :param inputs: (N, C, H, W)
    :param gt_boxes: (M, 5) [x1, y1, x2, y2, cls]
    :return: (N, H, W, C)
    """
    n,c, h, w = inputs.shape
    Is = []
    for i in range(n):
        img = inputs[i, :, :, :]
        if data_format == 'NCHW':
            img = np.transpose(img, [1, 2, 0])
        img = image_transform_inv(img)
        img = np.ascontiguousarray(img)

        gb = np.round(gt_boxes[i]).astype(np.int)
        scores = gb[:, 4]
        class_names = range(0, 81) if class_names is None else class_names
        img = vis.draw_detection(img, bboxes=gb[:, :4], cls_inds=gb[:, 4], cls_name=class_names,
                                 scores=scores)
        Is.append(img)

    return np.asarray(Is)

def draw_gtboxes_on_orignal_images(ori_images, gt_boxes, class_names=None):
    """
    Draw boxes on images
    :param ori_images: a list of N images (H, W, C)
    :param gt_boxes: (M, 5) [x1, y1, x2, y2, cls]
    :return: (N, H, W, C)
    """
    Is = []
    for i, img in enumerate(ori_images):

        h, w, _ = img.shape
        # img = (img + 1.0) * 128.0
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        gb = np.round(gt_boxes[i]).astype(np.int)
        gb[0:4:2] = gb[0:4:2] * w / cfg.input_size[1]
        gb[1:4:2] = gb[1:4:2] * h / cfg.input_size[0]
        scores = 0 - np.ones([gb.shape[0]])
        class_names = range(0, 81) if class_names is None else class_names
        img = vis.draw_detection(img, bboxes=gb[:, :4], cls_inds=gb[:, 4], cls_name=class_names,
                                 scores=scores)
        Is.append(img)

    return Is

def draw_anchors(inputs, Targets, ANCHORS, class_names):
    """
    Draw positive anchors on the images
    :param inputs: of shape (N, C, H, W), float images
    :param Targets: a list of (labels, bboxes, bboxes_weights)
                    labels of shape (N, H, W, A)
                    bboxes of shape (N, H, W, A x 4)
    :return: images (N, H, W, 3)
    """
    n, c, h, w = inputs.shape
    Is = []
    cnt = []
    labels, _, _, _ = Targets

    for i in range(n):
        img = inputs[i, :, :, :]
        img = np.transpose(img, [1, 2, 0])
        # img = (img + 1.0) * 128.0
        # img = img.astype(np.uint8)
        img = image_transform_inv(img)
        img = np.ascontiguousarray(img)

        labels = Targets[0][i].reshape([-1])
        anc = np.copy(ANCHORS)
        bboxes = anc.reshape([-1, 4])
        assert labels.shape[0] == bboxes.shape[0]
        spatial_dim = labels.shape[0]
        pos_inds = np.where(labels > 0.001)[0]
        num = len(pos_inds)
        labels = np.round(labels[pos_inds]).astype(np.int)
        bboxes = np.round(bboxes[pos_inds]).astype(np.int)
        scores = np.ones([bboxes.shape[0]])
        class_names = range(0, 81) if class_names is None else class_names
        img = vis.draw_detection(img, bboxes, cls_inds=labels, scores=scores, thick=1,
                                 cls_name=class_names, ellipse=False)
        # print (bboxes)

        Is.append(img)
        cnt.append(num)

    return np.asarray(Is), np.asarray(cnt)

def draw_anchors_with_scores(inputs, Targets, ANCHORS, Outputs, class_names):
    n, c, h, w = inputs.shape
    Is = []
    cnt = []

    anc = [out[0].reshape([-1, 4]) for out in Outputs]

    for i in range(n):
        img = inputs[i, :, :, :]
        img = np.transpose(img, [1, 2, 0])
        img = (img + 1.0) * 128.0
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        num = 0
        labels, _, _, _ = Targets
        if isinstance(ANCHORS, list):
            anc = [anc.reshape([-1, 4]) for anc in ANCHORS]
        anc = np.vstack(anc)

        labels = labels[i].reshape([-1])
        anc = np.copy(anc)
        bboxes = anc.reshape([-1, 4])
        assert labels.shape[0] == bboxes.shape[0]
        spatial_dim = labels.shape[0]
        pos_inds = np.where(labels > 0.001)[0]
        num += len(pos_inds)
        labels = np.round(labels[pos_inds]).astype(np.int)
        bboxes = np.round(bboxes[pos_inds]).astype(np.int)
        scores = np.ones([bboxes.shape[0]])
        class_names = range(0, 81) if class_names is None else class_names
        img = vis.draw_detection(img, bboxes, cls_inds=labels, scores=scores, thick=1,
                                 cls_name=class_names, ellipse=True)
        # print (bboxes)

        Is.append(img)
        cnt.append(num)

    return np.asarray(Is), np.asarray(cnt)

def draw_images(inputs):
    n, c, h, w = inputs.shape
    Is = []
    for i in range(n):
        img = inputs[i, :, :, :]
        img = np.transpose(img, [1, 2, 0])
        img = (img + 1.0) * 128.0
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)
        Is.append(img)
    return np.asarray(Is)

def dets_cluster(dets, keeps, clusters):
    centers = []
    for ind in keeps:
        inds = np.where(clusters == ind)[0]
        dets_in = dets[inds, :]
        if len(inds) > 1:
            score = dets_in[:, 4]
            dets_in[:, :4] = dets_in[:, :4] * score[:, np.newaxis] / np.sum(score)
            dets_in = np.sum(dets_in, axis=0)
            dets_in[4] = np.max(score)
        centers.append(dets_in)
    if len(centers) > 0:
        cls_dets = np.vstack(centers)
        return cls_dets
    return dets
