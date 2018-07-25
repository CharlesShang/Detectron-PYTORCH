#include <THC/THC.h>
#include <math.h>
#include "cuda/anchor_target_kernel.h"

extern THCState *state;

int anchor_target_forward_cuda(THCudaTensor * anchors, THCudaTensor * gt_boxes,
                            float bg_overlap, float fg_overlap, float ignored_overlap,
                            THCudaLongTensor * labels, THCudaTensor * deltas, THCudaTensor * bbwght,
                            THCudaTensor * overlaps)
{
    float * anchors_flat = THCudaTensor_data(state, anchors);
    float * gts_flat = THCudaTensor_data(state, gt_boxes);

    long * labels_flat = THCudaLongTensor_data(state, labels);
    float * deltas_flat = THCudaTensor_data(state, deltas);
    float * bbwght_flat = THCudaTensor_data(state, bbwght);
    float * overlaps_flat = THCudaTensor_data(state, overlaps);

    THArgCheck(THCudaTensor_isContiguous(state, anchors), 0, "anchors must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, gt_boxes), 1, "gt_boxes must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, labels), 5, "labels must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, deltas), 6, "deltas must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, bbwght), 7, "bbwght must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, overlaps), 8, "overlaps must be contiguous");

     // Number of ROIs
    int num_anchors = THCudaTensor_size(state, anchors, 0);
    int num_labels = THCudaLongTensor_size(state, labels, 0);
    int num_deltas = THCudaTensor_size(state, deltas, 0);
    int num_bbwghts = THCudaTensor_size(state, bbwght, 0);
    int num_overlaps = THCudaTensor_size(state, overlaps, 0);
    if (num_anchors != num_labels || num_labels != num_deltas ||
        num_deltas != num_bbwghts || num_anchors != num_overlaps)
    {
        fprintf(stderr, "anchors and labels should have the same number of rows");
        exit(-1);
        return 0;
    }
    int dim_anchors = THCudaTensor_size(state, anchors, 1);
    if (dim_anchors != 4)
    {
        fprintf(stderr, "anchors should have Nx4 dims (anchor_num, 4)");
        exit(-1);
        return 0;
    }
    int num_gts = THCudaTensor_size(state, gt_boxes, 0);
    int dim_gts = THCudaTensor_size(state, gt_boxes, 1);
    int dim_ovs_0 = THCudaTensor_size(state, overlaps, 0);
    int dim_ovs_1 = THCudaTensor_size(state, overlaps, 1);
    if (dim_gts != 5)
    {
        fprintf(stderr, "gt_boxes should have Gx5 dims (x1, x2, y1, y2, cls)");
        exit(-1);
        return 0;
    }
    if (dim_ovs_0 != num_anchors || dim_ovs_1 != num_gts){
        fprintf(stderr, "overlaps dimension wont match");
        exit(-1);
        return 0;
    }
    cudaStream_t stream = THCState_getCurrentStream(state);
    AnchorTargetForwardLaucher(anchors_flat,
                            gts_flat,
                            num_anchors, num_gts,
                            bg_overlap, fg_overlap, ignored_overlap,
                            labels_flat, deltas_flat, bbwght_flat, overlaps_flat, stream);
    return 1;
}