import torch
from torch import nn

from crop_and_resize import CropAndResizeFunction


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(RoIAlign, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, featuremap, rois, spatial_scale):
        """
        RoIAlign based on crop_and_resize.
        :param featuremap: NxCxHxW
        :param rois: Mx5 float box with (id, x1, y1, x2, y2) **without normalization**
        :param spatial_scale: a float, indicating the size ratio w.r.t. the original image
        :return: MxCxoHxoW
        """
        boxes = rois[:, 1:5].contiguous()
        box_ind = rois[:, 0].int().contiguous()
        boxes = float(spatial_scale) * boxes

        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)

        spacing_w = (x2 - x1) / float(self.crop_width)
        spacing_h = (y2 - y1) / float(self.crop_height)

        image_height, image_width = featuremap.size()[2:4]
        nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
        ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)

        nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
        nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)

        boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)

        return CropAndResizeFunction(self.crop_height, self.crop_width, 0)(featuremap, boxes, box_ind)
