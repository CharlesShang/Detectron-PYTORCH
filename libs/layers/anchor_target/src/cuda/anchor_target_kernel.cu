#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "anchor_target_kernel.h"


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

    __global__ void AnchorTargetForward(
        const float* anchors, const float* gt_boxes,
        const int num_anchors, const int num_gts,
        const float bg_overlap, const float fg_overlap, const float ignored_overlap,
        long* labels, float* deltas, float* bbwght, float* overlaps) {

        CUDA_1D_KERNEL_LOOP(index, num_anchors) {

            int n = index * 4;
            float best_iou = 0.;
            int best_ind = -1;
            int match_cls = -1;
            for (int j = 0; j < num_gts; j ++){
                int m = j * 5;
                int cls = int(gt_boxes[m+4]);
                float iou = devIoU(anchors + n, gt_boxes + m, cls);
                overlaps[index * num_gts + j] = iou;
                if (best_iou <= iou)
                {
                    best_iou = iou;
                    best_ind = j;
                    match_cls = cls;
                }
                if (cls <= 0 && iou >= ignored_overlap){
                    best_iou = iou;
                    match_cls = cls;
                    best_ind = j;
                    break;
                }
            }
            if(match_cls <= 0 && best_iou >= ignored_overlap){
                labels[index] = -1;
            }else if(match_cls > 0 && best_iou >= fg_overlap){
                labels[index] = match_cls;
            }else if(best_iou < fg_overlap && best_iou >= bg_overlap){
                labels[index] = -1;
            }else{
                labels[index] = 0;
            }
            if (best_iou >= fminf(0.5, fg_overlap) && match_cls > 0 && best_ind >= 0){
                bbox_encoding(anchors + n, gt_boxes + best_ind * 5, deltas + n);
                bbwght[n] = 1.0;
                bbwght[n + 1] = 1.0;
                bbwght[n + 2] = 1.0;
                bbwght[n + 3] = 1.0;
            }
        }
    }

    __global__ void AssignBestMatchKernel(
        const float* anchors, const float* gt_boxes,
        const int num_anchors, const int num_gts,
        long* labels, float* deltas, float* bbwght, float* overlaps) {

        CUDA_1D_KERNEL_LOOP(index, num_gts) {

            int n = index * 5;
            float best_iou = 0.;
            int best_ind = -1;
            int match_cls = gt_boxes[n + 4];
            if(match_cls > 0){
                for (int j = 0; j < num_anchors; j ++){
                    if (best_iou < overlaps[j * num_gts + index]){
                        best_iou = overlaps[j * num_gts + index];
                        best_ind = j;
                    }
                }
                labels[best_ind] = match_cls;
                int m = best_ind * 4;
                bbox_encoding(anchors + m, gt_boxes + n, deltas + m);
                bbwght[m] = 1.0;
                bbwght[m+1] = 1.0;
                bbwght[m+2] = 1.0;
                bbwght[m+3] = 1.0;
            }
        }
    }

    int AnchorTargetForwardLaucher(
        const float* anchors, const float* gt_boxes,
        const int num_anchors, const int num_gts,
        const float bg_overlap, const float fg_overlap, const float ignored_overlap,
        long* labels, float* deltas, float* bbwght, float* overlaps, cudaStream_t stream)
    {
        const int kThreadsPerBlock = 1024;
        cudaError_t err;
        AnchorTargetForward<<<(num_anchors + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            anchors, gt_boxes, num_anchors, num_gts,
            bg_overlap, fg_overlap, ignored_overlap,
            labels, deltas, bbwght, overlaps
        );
//        AssignBestMatchKernel<<<(num_gts + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
//            anchors, gt_boxes, num_anchors, num_gts,
//            labels, deltas, bbwght, overlaps
//        );

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


