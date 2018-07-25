import cv2
import numpy as np
from libs.boxes.bbox_transform import clip_boxes
from .sample_distorted_bounding_box import sample_distored_bounding_box

def imcv2_recolor(im, a=0.1):

    # t = np.random.uniform(-1, 1, 3)
    im = im.astype(np.float32)
    # im *= (1 + t * a)
    im = image_transform(im)

    # up = np.random.uniform(-1, 1)
    # im = np.power(im, 1. + up * .5)

    return im


def random_crop(im, masks, mask):
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    im = im[offy: (offy + h), offx: (offx + w)]
    mask = mask[offy: (offy + h), offx: (offx + w)]

    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)
        mask = cv2.flip(mask, 1)

    if masks.size > 0:
        masks = np.transpose(masks, (1, 2, 0)) # to (h, w, n)
        masks = cv2.resize(masks, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        masks = masks[offy: (offy + h), offx: (offx + w)]
        if flip:
            masks = cv2.flip(masks, 1)
        try:
            if masks.ndim > 2:
                masks = np.transpose(masks, (2, 0, 1))  # to (n, h, w)
            else:
                masks = masks.reshape((1, h, w))
        except ValueError:
            print (masks.ndim, masks.shape)
            raise
    else:
        masks = np.zeros((0, h, w), masks.dtype)

    return im, masks, mask, [scale, [offx, offy], flip]

def _aspect_ratio_boxes(boxes, fx=1., fy=1.):
    boxes[:, 0:4:2] = boxes[:, 0:4:2] * fx
    boxes[:, 1:4:2] = boxes[:, 1:4:2] * fy
    return boxes

def random_aspect_ratio(im, inst_masks, mask, boxes, classes, min_obj_cover=0.5,
                        min_aspect_ratio=0.667, max_aspect_ratio=1.667):
    """return an image with the same width, but a random height within [down, upper] * height"""
    im, bboxes, labels, inst_masks, mask = \
        sample_distored_bounding_box(im, bboxes=boxes, labels=classes, inst_masks=inst_masks, mask=mask,
                                     min_aspect_ratio=min_aspect_ratio, max_aspect_ratio=max_aspect_ratio,
                                     min_obj_cover=min_obj_cover)

    return im, inst_masks, mask, bboxes, labels

def random_flip(im, inst_masks, mask, boxes, classes):
    h, w, c = im.shape
    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)
        mask = cv2.flip(mask, 1)
    if inst_masks.size > 0:
        inst_masks = np.transpose(inst_masks, (1, 2, 0)) # to (h, w, n)
        if flip:
            inst_masks = cv2.flip(inst_masks, 1)
        try:
            if inst_masks.ndim > 2:
                inst_masks = np.transpose(inst_masks, (2, 0, 1))  # to (n, h, w)
            else:
                inst_masks = inst_masks.reshape((1, h, w))
        except ValueError:
            print (inst_masks.ndim, inst_masks.shape)
            raise
    else:
        inst_masks = np.zeros((0, h, w), inst_masks.dtype)

    boxes = _offset_boxes(boxes, im.shape, 1, [0, 0], flip)
    return im, inst_masks, mask, boxes, classes

def random_scale(im, inst_masks, mask, boxes, classes, scale):
    """Randomly scaling the image and corresponding annotations"""
    # scale = np.random.uniform(down, upper)
    h, w, c = im.shape
    if scale > 1:
        """"""
        max_offx = (scale - 1.) * w
        max_offy = (scale - 1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)
        im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        im = im[offy: (offy + h), offx: (offx + w)]
        mask = mask[offy: (offy + h), offx: (offx + w)]
        if inst_masks.size > 0:
            inst_masks = np.transpose(inst_masks, (1, 2, 0))  # to (h, w, n)
            inst_masks = cv2.resize(inst_masks, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            inst_masks = inst_masks[offy: (offy + h), offx: (offx + w)]
            try:
                if inst_masks.ndim > 2:
                    inst_masks = np.transpose(inst_masks, (2, 0, 1))  # to (n, h, w)
                else:
                    inst_masks = inst_masks.reshape((1, h, w))
            except ValueError:
                print (inst_masks.ndim, inst_masks.shape)
                raise
        else:
            inst_masks = np.zeros((0, h, w), inst_masks.dtype)
    else:
        """"""
        canvas = np.zeros(im.shape, im.dtype) + np.array([103, 116, 123], im.dtype)
        canvas_mask = np.zeros(mask.shape, mask.dtype)
        max_offx = (scale - 1.) * w
        max_offy = (scale - 1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)
        im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        h_, w_, _ = im.shape
        canvas[-offy: (-offy + h_), -offx: (-offx + w_)] = im
        canvas_mask[-offy: (-offy + h_), -offx: (-offx + w_)] = mask
        if inst_masks.size > 0:
            inst_masks = np.transpose(inst_masks, (1, 2, 0))  # to (h, w, n)
            canvas_instmask = np.zeros(inst_masks.shape, inst_masks.dtype)
            inst_masks = cv2.resize(inst_masks, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            if inst_masks.ndim == 2:
                inst_masks = inst_masks[:,:, np.newaxis]
            canvas_instmask[-offy: (-offy + h_), -offx: (-offx + w_)] = inst_masks
            canvas_instmask = np.transpose(canvas_instmask, (2, 0, 1))  # to (n, h, w)
        else:
            canvas_instmask = np.zeros((0, h, w), inst_masks.dtype)

        im, mask, inst_masks = canvas, canvas_mask, canvas_instmask

    boxes = _offset_boxes(boxes, im.shape, scale, [offx, offy], False)
    boxes, classes, inst_masks = _filter_invalid_boxes(boxes, classes, inst_masks, min_size=3)

    return im, inst_masks, mask, boxes, classes

def fixed_scale(im, inst_masks, mask, boxes, classes, target_h, target_w):
    """to fixed scale without and randomness, without keeping the aspect ratio"""
    h, w, c = im.shape
    scale = float(target_h) / h
    im = cv2.resize(im, (target_w, target_h))
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if inst_masks.size > 0:
        inst_masks = np.transpose(inst_masks, (1, 2, 0))  # to (h, w, n)
        inst_masks = cv2.resize(inst_masks, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        try:
            if inst_masks.ndim > 2:
                inst_masks = np.transpose(inst_masks, (2, 0, 1))  # to (n, h, w)
            else:
                inst_masks = inst_masks.reshape((1, target_h, target_w))
        except ValueError:
            print (inst_masks.ndim, inst_masks.shape)
            raise
    else:
        inst_masks = np.zeros((0, h, w), inst_masks.dtype)
    boxes[:, 0:4:2] = boxes[:, 0:4:2] * float(target_w) / w
    boxes[:, 1:4:2] = boxes[:, 1:4:2] * float(target_h) / h

    return im, inst_masks, mask, boxes, classes


def resize_as_min_side(im, masks, mask, boxes, classes, min_side, max_side):
    """resize image so that it may max-fix the canvas"""
    h, w, c = im.shape
    n = classes.size
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(min_side) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_side:
        im_scale = float(max_side) / float(im_size_max)

    new_w, new_h = int(im_scale * w), int(im_scale * h)
    im = cv2.resize(im, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    boxes *= im_scale
    if masks.size > 0:
        masks = np.transpose(masks, (1, 2, 0))  # to (h, w, n)
        masks = cv2.resize(masks, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if masks.ndim > 2:
            masks = np.transpose(masks, (2, 0, 1))  # to (n, h, w)
        else:
            masks = masks.reshape((1, new_h, new_w))

    return im, masks, mask, boxes, classes, im_scale


def pad_to_canvas(im, masks, mask, boxes, classes, canvas_width, canvas_height):
    h, w, c = im.shape
    assert h <= canvas_height and w <= canvas_width
    im_ = np.zeros((canvas_height, canvas_width, c), dtype=im.dtype)
    im_[:h, :w, :] = im
    mask_ = np.zeros((canvas_height, canvas_width), dtype=mask.dtype)
    mask_[:h, :w] = mask
    if masks.size > 0:
        n = masks.shape[0]
        masks_ = np.zeros((n, canvas_height, canvas_width), dtype=masks.dtype)
        masks_[:, :h, :w] = masks
    else:
        masks_ = masks
    return im_, masks_, mask_, boxes, classes


def center_crop2fixed_cut(im, masks, mask, boxes, classes, target_width, target_height, min_size=2):
    """drop some pixel on the longest side"""

    h, w, c = im.shape
    if float(target_width) / w > float(target_height) / h:
        new_w, new_h = int(target_width), int(float(target_width) / w * h)
    else:
        new_w, new_h = int(float(target_height) / h * w), int(target_height)

    scale = float(new_w) / w
    offset_w, offset_h = 0, 0
    if new_w - target_width + 1 > 0 and new_h - target_height + 1 > 0:
        offset_w = np.random.randint(0, new_w - target_width + 1)
        offset_h = np.random.randint(0, new_h - target_height + 1)
    # offset_w = int((new_w - target_width) / 2)
    # offset_h = int((new_h - target_height) / 2)

    im = cv2.resize(im, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    im = im[offset_h: (offset_h + target_height), offset_w: (offset_w + target_width)]
    mask = mask[offset_h: (offset_h + target_height), offset_w: (offset_w + target_width)]

    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)
        mask = cv2.flip(mask, 1)

    if masks.size > 0:
        masks = np.transpose(masks, (1, 2, 0)) # to (h, w, n)
        masks = cv2.resize(masks, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        masks = masks[offset_h: (offset_h + target_height), offset_w: (offset_w + target_width)]
        if flip:
            masks = cv2.flip(masks, 1)
        try:
            if masks.ndim > 2:
                masks = np.transpose(masks, (2, 0, 1))  # to (n, h, w)
            else:
                masks = masks.reshape((1, target_height, target_width))
        except ValueError:
            print (masks.ndim, masks.shape)
            raise
    else:
        masks = np.zeros((0, target_height, target_width), masks.dtype)

    # bboxes
    boxes = _offset_boxes(boxes, [target_height, target_width], scale, [offset_w, offset_h], flip)
    # boxes *= scale
    # boxes = clip_boxes(boxes, [target_height, target_width])
    # if flip:
    #     boxes_x = np.copy(boxes[:, 0])
    #     boxes[:, 0] = target_width - boxes[:, 2]
    #     boxes[:, 2] = target_width - boxes_x

    boxes, classes, masks = _filter_invalid_boxes(boxes, classes, masks, min_size=min_size)

    return im, masks, mask, boxes, classes

def center_crop2fixed_pad(im, masks, mask, boxes, classes, target_width, target_height, min_size=2):
    """padding zeros on the shortest side"""

    h, w, c = im.shape
    ir, tr = float(h) / w, float(target_height) / target_width
    if ir > tr:
        borderw, borderh = int((h / tr - w) / 2), 0
    else:
        borderh, borderw = int((w * tr - h) / 2), 0

    im = cv2.copyMakeBorder(im, borderh, borderh, borderw, borderw, cv2.BORDER_CONSTANT, value=[103, 116, 123])
    mask = cv2.copyMakeBorder(mask, borderh, borderh, borderw, borderw, cv2.BORDER_CONSTANT, value=[0])
    n = masks.shape[0]
    if n > 1:
        masks = [cv2.copyMakeBorder(m, borderh, borderh, borderw, borderw, cv2.BORDER_CONSTANT, value=[0]) for m in masks]
        masks = np.asarray(masks)
    elif n == 1:
        masks = cv2.copyMakeBorder(masks.reshape([h, w]), borderh, borderh, borderw, borderw, cv2.BORDER_CONSTANT, value=[0])
        masks = masks[np.newaxis, :, :]

    boxes[:, 0] = boxes[:, 0] + borderw
    boxes[:, 1] = boxes[:, 1] + borderh
    boxes[:, 2] = boxes[:, 2] + borderw
    boxes[:, 3] = boxes[:, 3] + borderh

    scale = float(target_height) / im.shape[0]
    im = cv2.resize(im, (target_width, target_height))
    mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)
        mask = cv2.flip(mask, 1)

    if masks.size > 0:
        masks = np.transpose(masks, (1, 2, 0)) # to (h, w, n)
        masks = cv2.resize(masks, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        if flip:
            masks = cv2.flip(masks, 1)
        try:
            if masks.ndim > 2:
                masks = np.transpose(masks, (2, 0, 1))  # to (n, h, w)
            else:
                masks = masks.reshape((1, target_height, target_width))
        except ValueError:
            print (masks.ndim, masks.shape)
            raise
    else:
        masks = np.zeros((0, target_height, target_width), masks.dtype)

    # bboxes
    boxes = _offset_boxes(boxes, [target_height, target_width], scale, [0, 0], flip)
    boxes, classes, masks = _filter_invalid_boxes(boxes, classes, masks, min_size=min_size)
    return im, masks, mask, boxes, classes

def _offset_boxes(boxes, im_shape, scale, offs, flip):
    if len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float)
    boxes *= scale
    boxes[:, 0::2] -= offs[0]
    boxes[:, 1::2] -= offs[1]
    boxes = clip_boxes(boxes, im_shape)

    if flip:
        boxes_x = np.copy(boxes[:, 0])
        boxes[:, 0] = im_shape[1] - boxes[:, 2]
        boxes[:, 2] = im_shape[1] - boxes_x
    return boxes


def _filter_invalid_boxes(boxes, classes, masks, min_size=4):
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep_inds = np.where(np.logical_and(ws > min_size, hs > min_size))[0]
    boxes, classes, masks = boxes[keep_inds], classes[keep_inds], masks[keep_inds]
    return boxes, classes, masks

def image_transform(im, format='pytorch'):
    """BGR [0-255] to RGB [0, 1] then
     using mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] to normalize images"""
    if format == 'pytorch':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32)
        im = im / 255.
        im = (im - np.array([0.485, 0.456, 0.406], dtype=np.float32) ) / \
                        np.array([0.229, 0.224, 0.225], dtype=np.float32)
    elif format == 'caffe':
        # return BGR image
        im = im.astype(np.float32)
        im = im - np.array([103.939, 116.779, 123.68], dtype=np.float32)
    return im


def image_transform_inv(im, format='pytorch'):
    """INVERSE:
    BGR [0-255] to RGB [0, 1] then
    using mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] to normalize images"""
    if format == 'pytorch':
        im = im * np.array([0.229, 0.224, 0.225], dtype=np.float32) + \
             np.array([0.485, 0.456, 0.406], dtype=np.float32)
        im = im * 255.
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif format == 'caffe':
        im = im + np.array([103.939, 116.779, 123.68], dtype=np.float32)
    return im.astype(np.uint8)

def distort_color(im):
    # distort brightness
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    v_ += np.random.randint(-16, 16)
    v_[v_ > 255] = 255
    v_[v_ < 0] = 0
    hsv = cv2.merge((h_, s_, v_))
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # distort contrast
    # """
    # From TF source code
    #   For each channel, this Op computes the mean of the image pixels in the
    # channel and then adjusts each component `x` of each pixel to
    # `(x - mean) * contrast_factor + mean`.
    # """
    im = im.astype(np.float32)
    b, g, r = cv2.split(im)
    # factor = (np.random.rand() + 0.5)
    factor = np.random.uniform(0.75, 1.25)
    b = (b - b.mean()) * factor + b.mean()
    b[b > 255] = 255
    b[b < 0] = 0
    # factor = (np.random.rand() + 0.5)
    g = (g - g.mean()) * factor + g.mean()
    g[g > 255] = 255
    g[g < 0] = 0
    # factor = (np.random.rand() + 0.5)
    r = (r - r.mean()) * factor + r.mean()
    r[r > 255] = 255
    r[r < 0] = 0
    im = cv2.merge((b, g, r))

    # im = im.astype(np.uint8)
    # clip_value = np.random.rand() * 3.0
    # clahe = cv2.createCLAHE(clipLimit=clip_value, tileGridSize=(8, 8))
    # lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    # l, a, b = cv2.split(lab)  # split on 3 different channels
    # l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    # lab = cv2.merge((l2, a, b))  # merge channels
    # im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    # im = im.astype(np.float32)

    return im


def preprocess_train_keep_aspect_ratio(im, boxes, classes, inst_masks, mask, min_side, max_side,
                                       canvas_height, canvas_width,
                                       use_augment=False, training_scale=[0.3, 0.5, 0.7, 1.0]):
    """ Preprocessing images, boxes, classes, etc.
    Params:
    im: of shape (H, W, 3)
    boxes: of shape (N, 4)
    classes: of shape (N,)
    masks: of shape (N, H, W)
    mask: of shape (H, W)
    input_size: a list specify input shape (ih, iw)

    Return:
    """
    im, inst_masks, mask, boxes, classes, im_scale = resize_as_min_side(im, inst_masks, mask, boxes, classes,
                                                                        min_side=min_side, max_side=max_side)

    im, inst_masks, mask, boxes, classes = random_flip(im, inst_masks, mask, boxes, classes)
    if use_augment:
        if np.random.choice([0, 1]) != 0:
            scale = np.random.choice(training_scale)  # adding more small objects
            im, inst_masks, mask, boxes, classes = random_scale(im, inst_masks, mask, boxes, classes, scale=scale)

    im, inst_masks, mask, boxes, classes = pad_to_canvas(im, inst_masks, mask, boxes, classes,
                                                         canvas_width=canvas_width,
                                                         canvas_height=canvas_height)

    # im = distort_color(im)
    im = imcv2_recolor(im)

    boxes = np.asarray(boxes, dtype=np.float32)
    inst_masks = np.zeros([1, im.shape[0], im.shape[1]], dtype=inst_masks.dtype) if inst_masks.size == 0 else inst_masks
    return im, boxes, classes, inst_masks, mask, im_scale


def preprocess_eval_keep_aspect_ratio(im, boxes, classes, inst_masks, mask, min_side, max_side):
    """ Preprocessing images, boxes, classes, etc.
    Params:
    im: of shape (H, W, 3)
    boxes: of shape (N, 4)
    classes: of shape (N,)
    masks: of shape (N, H, W)
    mask: of shape (H, W)
    input_size: a list specify input shape (ih, iw)

    Return:
    """
    im, inst_masks, mask, boxes, classes, im_scale = resize_as_min_side(im, inst_masks, mask, boxes, classes,
                                                                        min_side=min_side, max_side=max_side)
    im = imcv2_recolor(im)

    boxes = np.asarray(boxes, dtype=np.float32)
    inst_masks = np.zeros([1, im.shape[0], im.shape[1]], dtype=inst_masks.dtype) if inst_masks.size == 0 else inst_masks
    return im, boxes, classes, inst_masks, mask, im_scale


def preprocess_test_keep_aspect_ratio(im, min_side, max_side):
    ori_im = np.copy(im)
    im = image_transform(im)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(min_side) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_side:
        im_scale = float(max_side) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def preprocess_train(im, boxes, classes, inst_masks, mask, input_size, min_size=2,
                     use_augment=False, training_scale=[0.3, 0.5, 0.7, 1.0]):
    """ Preprocessing images, boxes, classes, etc.
    Params:
    im: of shape (H, W, 3)
    boxes: of shape (N, 4)
    classes: of shape (N,)
    masks: of shape (N, H, W)
    mask: of shape (H, W)
    input_size: a list specify input shape (ih, iw)
    
    Return: 
    """
    ori_im = np.copy(im)
    target_h, target_w = input_size

    # ---------- old data_augmentation ----------
    if use_augment:
        if np.random.choice([0, 1]) != 0:
            scale = np.random.choice(training_scale) # adding more small objects
            im, inst_masks, mask, boxes, classes = random_scale(im, inst_masks, mask, boxes, classes, scale=scale)
            min_obj_cover = np.random.choice([0.8, 0.9, 1.0])
            # truncted examples may lead to multiple-detections..
            im, inst_masks, mask, boxes, classes = random_aspect_ratio(im, inst_masks, mask, boxes, classes,
                                                                       min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                                                                       min_obj_cover=min_obj_cover)
    #
    # # r = np.random.randint(0, 3)
    # if np.random.rand() < 0.75:
    #     im, inst_masks, mask, boxes, classes = fixed_scale(im, inst_masks, mask, boxes, classes, target_h, target_w)
    # else:
    #     im, inst_masks, mask, boxes, classes = center_crop2fixed_pad(im, inst_masks, mask, boxes, classes, target_w, target_h,
    #                                                                  min_size=min_size)

    # ---------- old data_augmentation ----------

    # ---------- none data_augmentation ----------
    im, inst_masks, mask, boxes, classes = fixed_scale(im, inst_masks, mask, boxes, classes, target_h, target_w)
    im, inst_masks, mask, boxes, classes = random_flip(im, inst_masks, mask, boxes, classes)
    # ---------- none data_augmentation ----------

    # ---------- old data_augmentation ----------
    im = distort_color(im)
    # ---------- old data_augmentation ----------

    im = imcv2_recolor(im)

    # add this because zeros numpy array will cause errors in torch Dataloader
    inst_masks = np.zeros([1, target_h, target_w], dtype=inst_masks.dtype) if inst_masks.size == 0 else inst_masks

    boxes = np.asarray(boxes, dtype=np.float32)
    return im, boxes, classes, inst_masks, mask, ori_im

def preprocess_test(im, input_size):

    ori_im = np.copy(im)

    h, w = input_size
    im = cv2.resize(im, (w, h))
    im = image_transform(im)

    return im, ori_im