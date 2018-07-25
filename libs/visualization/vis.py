import cv2
import numpy as np
import PIL.ImageColor as ImageColor


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Crimson',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'MistyRose', 'OliveDrab', 'Cornsilk', 'Cyan', 'Violet',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Wheat', 'White', 'Coral',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'Beige', 'Bisque', 'CornflowerBlue'
]
STANDARD_COLORS = [
    'darkblue', 'aqua', 'blueviolet', 'brown', 'chocolate', 'darkcyan', 'darkgreen', 'darkmagenta',
    'darkolivegreen', 'darkturquoise', 'deeppink', 'deepskyblue', 'dodgerblue', 'gold', 'indigo',
    'lawngreen', 'lightseagreen', 'limegreen', 'magenta', 'olive', 'orange', 'purple', 'seagreen',
    'violet', 'yellowgreen', 'tomato', 'sienna',
] + STANDARD_COLORS
# STANDARD_COLORS.sort()

# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return r * 127, g * 127, b * 127


# def get_color(indx, cls_num=-1):
#     if indx < 0:
#         return (255, 255, 255)
#     if indx >= cls_num:
#         return (23 * indx % 255, 47 * indx % 255, 137 * indx % 255)
#     base = int(np.ceil(pow(cls_num, 1. / 3)))
#     return _to_color(indx, base)

def get_color(indx, cls_num=-1):
    return ImageColor.getrgb(STANDARD_COLORS[indx])[::-1] # BGR


def draw_detection(im, bboxes, scores=None, cls_inds=None, cls_name=None, color=None, thick=None, ellipse=False):
    # draw image
    bboxes = np.round(bboxes).astype(np.int)
    if cls_inds is not None:
        cls_inds = cls_inds.astype(np.int)
    cls_num = len(cls_name) if cls_name is not None else -1

    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        cls_indx = cls_inds[i] if cls_inds is not None else None
        color_ = get_color(cls_indx, cls_num) if color == None else color
        color_ = (0, 0, 0) if cls_indx < 0 else color_

        thick = int((h + w) / 500) if thick == None else thick
        if not ellipse:
            cv2.rectangle(imgcv,
                          (box[0], box[1]), (box[2], box[3]),
                          color_, thick)
        else:
            cv2.ellipse(imgcv,
                        (box[0]/2 + box[2]/2, box[1]/2 + box[3]/2),
                        (box[2]/2 - box[0]/2, box[3]/2 - box[1]/2),
                        0, 0, 360,
                        color=color_, thickness=thick)

        if cls_indx is not None:
            score = scores[i] if scores is not None else 1
            name = cls_name[cls_indx] if cls_name is not None else str(cls_indx)
            name = 'ign' if cls_indx < 0 else name
            mess = '%s: %.2f' % (name[:4], score)
            cv2.putText(imgcv, mess, (box[0], box[1] - 8),
                        0, 1e-3 * h, color_, thick // 3)

    return imgcv