#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_target_kernel.h"


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


    __device__ inline float devIoU(float const * const a, float const * const b, const int cls) {
        float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
        float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
        float width = fmaxf(right - left + 1, 0.f), height = fmaxf(bottom - top + 1, 0.f);
        float interS = width * height;
        float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
        float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
        if (cls <= 0){
            return interS / Sa;
        }
        return interS / (Sa + Sb - interS);
    }

    __device__ inline void bbox_encoding(float const * const a, float const * const b, float* c) {
        // a: given box
        // b: grouth truth
        // c: output deltas
        const float bw = b[2] - b[0] + 1.0;
        const float bh = b[3] - b[1] + 1.0;
        const float bx = b[0] + bw * 0.5;
        const float by = b[1] + bh * 0.5;

        const float aw = a[2] - a[0] + 1.0;
        const float ah = a[3] - a[1] + 1.0;
        const float ax = a[0] + aw * 0.5;
        const float ay = a[1] + ah * 0.5;

        c[0] = (bx - ax) / aw / 0.1;
        c[1] = (by - ay) / ah / 0.1;
        c[2] = log(bw / aw) / 0.2;
        c[3] = log(bh / ah) / 0.2;
    }

    __global__ void ROITargetForward(
        const float* rois, const long* roi_batch_inds,
        const float* gt_boxes, const long* gt_batch_inds,
        const float bg_overlap, const float fg_overlap,
        const int num_rois, const int num_gts,
        long* labels, float* deltas, float* bbwght) {

        CUDA_1D_KERNEL_LOOP(index, num_rois) {

            int n = index * 4;
            int r_id = int(roi_batch_inds[index]);

            float best_iou = 0.;
            int best_ind = -1;
            int match_cls = -1;
            for (int j = 0; j < num_gts; j ++){
                int m = j * 5;
                int g_id = int(gt_batch_inds[j]);
                if (r_id != g_id) continue;
                int cls = int(gt_boxes[m+4]);
                float iou = devIoU(rois + n, gt_boxes + m, cls);
                if (best_iou <= iou)
                {
                    best_iou = iou;
                    best_ind = j;
                    match_cls = cls;
                }
            }
            if (best_iou >= fg_overlap){
                labels[index] = match_cls > 0? match_cls : -1;
            }else{
                labels[index] = 0;
            }
            if (best_iou >= fminf(0.5, fg_overlap) && match_cls > 0 && best_ind >= 0){
                bbox_encoding(rois + n, gt_boxes + best_ind * 5, deltas + n);
                bbwght[n] = 1.0;
                bbwght[n + 1] = 1.0;
                bbwght[n + 2] = 1.0;
                bbwght[n + 3] = 1.0;
            }
        }
    }

    int ROITargetForwardLaucher(
        const float* rois, const long* roi_batch_inds,
        const float* gt_boxes, const long* gt_batch_inds,
        const float bg_overlap, const float fg_overlap,
        const int num_rois, const int num_gts,
        long* labels, float* deltas, float* bbwght, cudaStream_t stream)
    {
        const int kThreadsPerBlock = 1024;
        cudaError_t err;
        ROITargetForward<<<(num_rois + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            rois, roi_batch_inds, gt_boxes, gt_batch_inds,
            bg_overlap, fg_overlap,
            num_rois, num_gts,
            labels, deltas, bbwght
        );

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        return 1;
    }

#ifdef __cplusplus
}
#endif


