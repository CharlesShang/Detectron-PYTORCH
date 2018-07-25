int anchor_target_forward_cuda(THCudaTensor * anchors, THCudaTensor * gt_boxes,
                            float bg_overlap, float fg_overlap, float ignored_overlap,
                            THCudaLongTensor * labels, THCudaTensor * deltas, THCudaTensor * bbwght,
                            THCudaTensor * overlaps);