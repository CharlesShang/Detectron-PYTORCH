import cv2
import numpy as np

from libs.boxes.bbox_transform import clip_boxes


def _factor_closest(num, factor):
    num = int(np.ceil(float(num) / factor) * factor)
    return num

def _im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def im_preprocess(im, inp_size=(600,), max_size=1000):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
        inp_size (tuple): (600,)
        max_size (int): 1000
    Returns:
        im_data (ndarray): a data blob holding an image pyramid
        im_info (ndarray): [ [ h, w, scale] ]
    """
    im_orig = im.astype(np.float32, copy=True)
    # im_orig -= PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_xfactors = []
    im_scale_yfactors = []

    for target_size in inp_size:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        # factor 32
        im_scale_x = _factor_closest(im_scale * im_shape[1], 32) / float(im_shape[1])
        im_scale_y = _factor_closest(im_scale * im_shape[0], 32) / float(im_shape[0])

        im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_xfactors.append(im_scale_x)
        im_scale_yfactors.append(im_scale_y)
        processed_ims.append(im)

    # Create a blob to hold the input images
    im_data = _im_list_to_blob(processed_ims)
    # im_scales = np.array(im_scale_factors)
    im_info = np.array(
        [
            [im_data.shape[1], im_data.shape[2], im_scale_yfactors[i], im_scale_xfactors[i]]
            for i in range(len(im_scale_xfactors))
            ],
        dtype=np.float)

    return im_data, im_info


def imcv2_recolor(im, a=.1):
    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    return np.array(im * 255., np.uint8)


def imcv2_affine_trans(im):
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    im = im[offy: (offy + h), offx: (offx + w)]
    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)

    return im, [scale, [offx, offy], flip]


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


def preprocess_train(im, boxes):
    im, trans_param = imcv2_affine_trans(im)
    scale, offs, flip = trans_param

    boxes = _offset_boxes(boxes, im.shape, scale, offs, flip)
    # boxes = np.asarray(boxes, dtype=np.int)

    im = imcv2_recolor(im)

    return im, boxes