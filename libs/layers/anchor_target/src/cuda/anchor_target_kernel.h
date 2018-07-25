#ifndef _ROI_TARGET_KERNEL
#define _ROI_TARGET_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

__global__ void AnchorTargetForward(
    const float* anchors, const float* gt_boxes,
    const int num_anchors, const int num_gts,
    const float bg_overlap, const float fg_overlap, const float ignored_overlap,
    long* labels, float* deltas, float* bbwght, float* overlaps);

__global__ void AssignBestMatchKernel(
    const float* anchors, const float* gt_boxes,
    const int num_anchors, const int num_gts,
    long* labels, float* deltas, float* bbwght, float* overlaps);

int AnchorTargetForwardLaucher(
    const float* anchors, const float* gt_boxes,
    const int num_anchors, const int num_gts,
    const float bg_overlap, const float fg_overlap, const float ignored_overlap,
    long* labels, float* deltas, float* bbwght, float* overlaps, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif

