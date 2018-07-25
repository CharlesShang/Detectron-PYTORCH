int roi_target_forward_cuda(THCudaTensor * rois, THCudaLongTensor * roi_batch_inds,
                            THCudaTensor * gt_boxes, THCudaLongTensor * gt_batch_inds,
                            float bg_overlap, float fg_overlap,
                            THCudaLongTensor * labels, THCudaTensor * deltas, THCudaTensor * bbwght);