#ifndef _ROI_TARGET_KERNEL
#define _ROI_TARGET_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

__global__ void ROITargetForward(
    const float* rois, const long* roi_batch_inds,
    const float* gt_boxes, const long* gt_batch_inds,
    const float bg_overlap, const float fg_overlap,
    const int num_rois, const int num_gts,
    long* labels, float* deltas, float* bbwght);

int ROITargetForwardLaucher(
    const float* rois, const long* roi_batch_inds,
    const float* gt_boxes, const long* gt_batch_inds,
    const float bg_overlap, const float fg_overlap,
    const int num_rois, const int num_gts,
    long* labels, float* deltas, float* bbwght, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif

