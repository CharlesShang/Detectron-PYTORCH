#include <THC/THC.h>
#include <math.h>
#include "cuda/roi_target_kernel.h"

extern THCState *state;

int roi_target_forward_cuda(THCudaTensor * rois, THCudaLongTensor * roi_batch_inds,
    THCudaTensor * gt_boxes, THCudaLongTensor * gt_batch_inds,
    float bg_overlap, float fg_overlap,
    THCudaLongTensor * labels, THCudaTensor * deltas, THCudaTensor * bbwght)
{
    float * rois_flat = THCudaTensor_data(state, rois);
    float * gts_flat = THCudaTensor_data(state, gt_boxes);
    long * roi_batch_inds_flat = THCudaLongTensor_data(state, roi_batch_inds);
    long * gt_batch_inds_flat = THCudaLongTensor_data(state, gt_batch_inds);

    long * labels_flat = THCudaLongTensor_data(state, labels);
    float * deltas_flat = THCudaTensor_data(state, deltas);
    float * bbwght_flat = THCudaTensor_data(state, bbwght);

    THArgCheck(THCudaTensor_isContiguous(state, rois), 0, "rois must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, roi_batch_inds), 1, "roi_batch_inds must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, gt_boxes), 2, "gt_boxes must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, gt_batch_inds), 3, "gt_batch_inds must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, labels), 6, "labels must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, deltas), 7, "deltas must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, bbwght), 8, "bbwght must be contiguous");

     // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int num_roi_inds = THCudaLongTensor_size(state, roi_batch_inds, 0);
    int num_labels = THCudaLongTensor_size(state, labels, 0);
    int num_deltas = THCudaLongTensor_size(state, deltas, 0);
    int num_bbwght = THCudaLongTensor_size(state, bbwght, 0);
    if (num_roi_inds != num_rois && num_labels == num_rois && num_deltas == num_rois && num_bbwght == num_rois)
    {
        fprintf(stderr, "roi_batch_inds and rois should have the same number of rows");
        exit(-1);
        return 0;
    }
    int dim_rois = THCudaTensor_size(state, rois, 1);
    if (dim_rois != 4)
    {
        fprintf(stderr, "rois should have Nx5 dims (batch_inds, x1, x2, y1, y2)");
        exit(-1);
        return 0;
    }
    int num_gts = THCudaTensor_size(state, gt_boxes, 0);
    int num_gt_inds = THCudaLongTensor_size(state, gt_batch_inds, 0);
    if (num_gt_inds != num_gts)
    {
        fprintf(stderr, "gt_batch_inds and gt_boxes should have the same number of rows");
        exit(-1);
        return 0;
    }
    int dim_gts = THCudaTensor_size(state, gt_boxes, 1);
    if (dim_gts != 5)
    {
        fprintf(stderr, "gt_boxes should have Gx5 dims (x1, x2, y1, y2, cls)");
        exit(-1);
        return 0;
    }
    cudaStream_t stream = THCState_getCurrentStream(state);
    ROITargetForwardLaucher(rois_flat, roi_batch_inds_flat,
                            gts_flat, gt_batch_inds_flat,
                            bg_overlap, fg_overlap, num_rois, num_gts,
                            labels_flat, deltas_flat, bbwght_flat, stream);
    return 1;
}