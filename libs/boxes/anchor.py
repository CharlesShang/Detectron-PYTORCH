from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from libs.boxes import cython_anchor


def anchor_pyramid_old(strides, ih, iw, anchor_scales, anchor_ratios, anchor_base,
                   anchor_shift=[[0.0, 0.0]]):
    """build anchors in a pyramid"""
    anchor_list = []
    for i, stride in enumerate(strides):
        height, width = int(ih / stride), int(iw / stride)
        scales = anchor_scales[i] if isinstance(anchor_scales[i], list) else anchor_scales
        all_anchors = anchors_plane(height, width, stride,
                                    scales=scales,
                                    ratios=anchor_ratios,
                                    base=anchor_base,
                                    anchor_shift=anchor_shift)
        anchor_list.append(all_anchors)

    all_anchors = []
    for i in range(len(anchor_list)):
        all_anchors.append(anchor_list[i].reshape((-1, 4)))
    all_anchors = np.vstack(all_anchors)
    return all_anchors


def anchor_pyramid(strides, ih, iw, anchor_scales, anchor_ratios, anchor_base,
                   anchor_shift=[[0.0, 0.0]]):
    """build anchors in a pyramid"""
    anchor_list = []
    for i, stride in enumerate(strides):
        height, width = int(ih / stride), int(iw / stride)
        scales = anchor_scales[i] if isinstance(anchor_scales[i], list) else anchor_scales
        all_anchors = anchor_one_plane(height, width, stride,
                                       anchor_scales=scales,
                                       anchor_ratios=anchor_ratios,
                                       anchor_base=anchor_base,
                                       anchor_shifts=anchor_shift)
        anchor_list.append(all_anchors)
    all_anchors = np.vstack(anchor_list)
    return all_anchors


def anchor_one_plane(height, width, stride,
                     anchor_scales=[8., 16., 32],
                     anchor_ratios=[0.5, 1.0, 2.0],
                     anchor_base=16,
                     anchor_shifts=[[0.0, 0.0]]):
    # build anchor for one position
    anc = anchors(anchor_scales, anchor_ratios, anchor_base).astype(np.float)
    anchor_shifts = np.array(anchor_shifts, dtype=np.float)
    anchor_shifts = np.hstack((anchor_shifts, anchor_shifts)) * stride
    anc_all = [anc + shift for shift in anchor_shifts]
    anc = np.vstack(anc_all)

    # build anchor for all positions
    shift_x = np.arange(0, width) * stride
    shift_y = np.arange(0, height) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # of shape (n, 4)
    locations = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    K = locations.shape[0]
    A = anc.shape[0]
    all_anchors = anc.reshape((1, A, 4)) + locations.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors


def anchor_one_plane2(height, width, stride,
                     anchor_scales=[8., 16., 32],
                     anchor_ratios=[0.5, 1.0, 2.0],
                     anchor_base=16,
                     anchor_shifts=[[0.0, 0.0]]):
    # build anchor for one position
    anc = anchors(anchor_scales, anchor_ratios, anchor_base).astype(np.float)
    anchor_shifts = np.array(anchor_shifts, dtype=np.float)
    anchor_shifts = np.hstack((anchor_shifts, anchor_shifts)) * stride
    anc_all = [anc + shift for shift in anchor_shifts]
    anc = anc.reshape((1, anc.shape[0], -4)) + anchor_shifts.reshape((anchor_shifts.shape[0], 1, 4))
    anc = anc.reshape((-1, 4))

    # build anchor for all positions
    shift_x = np.arange(0, width) * stride
    shift_y = np.arange(0, height) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # of shape (n, 4)
    locations = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    K = locations.shape[0]
    A = anc.shape[0]
    all_anchors = anc.reshape((1, A, 4)) + locations.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors



def anchors(scales=[2., 4., 8., 16., 32.], ratios=[0.5, 1, 2.0], base=16):
    """Get a set of anchors at one position """
    return generate_anchors(base_size=base, scales=np.asarray(scales, np.float32), ratios=ratios)


def anchors_plane(height, width, stride=1.0,
                  scales=[2., 4., 8., 16., 32.], ratios=[0.5, 1, 2.0], base=16,
                  anchor_shift=[[0.0, 0.0]]):
    """Get a complete set of anchors in a spatial plane,
  height, width are plane dimensions
  stride is scale ratio of
  """
    # TODO: implement in C, or pre-compute them, or set to a fixed input-shape
    # enum all anchors in a plane
    # scales = kwargs.setdefault('scales', [2, 4, 8, 16, 32])
    # ratios = kwargs.setdefault('ratios', [0.5, 1, 2.0])
    # base = kwargs.setdefault('base', 16)
    anc = anchors(scales, ratios, base).astype(np.float)
    anchor_shift = np.array(anchor_shift, dtype=np.float)
    all_anchors = cython_anchor.anchors_plane(height, width, stride, anc, anchor_shift)
    return all_anchors


# Written by Ross Girshick and Sean Bell
def generate_anchors(base_size, ratios, scales):
    """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """

    # base_anchor = np.array([1, 1, base_size, base_size]) - 1
    base_anchor = np.array([-base_size // 2, -base_size // 2, base_size // 2, base_size // 2], dtype=np.float)
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
  Return width, height, x center, and y center for an anchor (window).
  """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
  Enumerate a set of anchors for each scale wrt an anchor.
  """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


if __name__ == '__main__':
    import time

    t = time.time()
    a = anchors()
    num_anchors = 0

    # all_anchors = anchors_plane(200, 250, stride=4, boarder=0)
    # num_anchors += all_anchors.shape[0]
    anchor_shift = np.array([[0, 0]], dtype=np.float)
    for i in range(1):
        ancs = anchors()
        ancs = ancs.astype(np.float)
        all_anchors = cython_anchor.anchors_plane(200, 250, 4, ancs, anchor_shift)
        num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
        all_anchors = cython_anchor.anchors_plane(100, 125, 8, ancs, anchor_shift)
        num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
        all_anchors = cython_anchor.anchors_plane(50, 63, 16, ancs, anchor_shift)
        num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
        all_anchors = cython_anchor.anchors_plane(25, 32, 32, ancs, anchor_shift)
        num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
    print('average time: %f' % ((time.time() - t) / 10))
    print('anchors: %d' % (num_anchors / 10))
    print(a.shape, '\n', a)
    print(a.shape, '\n', all_anchors.reshape(-1, 4)[:30, :])
    print(all_anchors.shape)

    anchor_scales, anchor_ratios, anchor_base = [2, 4, 8, 16, 32], [0.5, 1, 2], 16,
    anchor_shift = [[0, 0], [0.5, 0]]

    t1 = time.time()
    anc1 = anchor_pyramid([4, 8, 16, 32], 1280, 640, anchor_scales, anchor_ratios, anchor_base, anchor_shift)
    t2 = time.time()
    anc2 = anchor_pyramid_old([4, 8, 16, 32], 1280, 640, anchor_scales, anchor_ratios, anchor_base, anchor_shift)
    t3 = time.time()
    print('time1 %f, time2 %f' % (t2 - t1, t3 - t2))
    print((anc1 - anc2).max(), (anc1 - anc2).min())
    
    a1 = anchor_one_plane2(10, 5, 4, anchor_scales, anchor_ratios, anchor_base, anchor_shift)
    a2 = anchor_one_plane(10, 5, 4, anchor_scales, anchor_ratios, anchor_base, anchor_shift)
    print((a1 - a2).max(), (a1 - a2).min())
    print(np.hstack((a1, a2)))
    # from IPython import embed
    # embed()
