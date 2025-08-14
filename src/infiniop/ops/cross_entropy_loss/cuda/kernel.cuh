#ifndef __CROSS_ENTROPY_KERNEL_CUH__
#define __CROSS_ENTROPY_KERNEL_CUH__

#include <hpcc_fp16.h> 
#include <math.h>    

__device__ __forceinline__ float to_float(float val) {
    return val;
}

__device__ __forceinline__ float to_float(__half val) {
    return __half2float(val);  // 来自 <hpcc_fp16.h>
}

__device__ __forceinline__ float to_float(__hpcc_bfloat16 val) {
    return __bfloat162float(val);  // 来自 <hpcc_fp16.h>
}


template <typename T>
__global__ void cross_entropy_loss_kernel(
    T* loss, const T* logits, const int* target,
    int N, int C, int inner_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * inner_size;
    if (idx >= total) return;

    int n = idx / inner_size;
    int inner = idx % inner_size;
    int tgt = target[idx];

    float max_val = -1e-12f; 
    for (int c = 0; c < C; ++c) {
        int offset = ((n * C + c) * inner_size) + inner;
        max_val = fmaxf(max_val, to_float(logits[offset]));
    }

    float sum_exp = 0.f;
    for (int c = 0; c < C; ++c) {
        int offset = ((n * C + c) * inner_size) + inner;
        sum_exp += expf(to_float(logits[offset]) - max_val);
    }

    int target_offset = ((n * C + tgt) * inner_size) + inner;
    float logit_tgt = to_float(logits[target_offset]);
    float prob = expf(logit_tgt - max_val) / sum_exp;
    float loss_val = -logf(prob);
    loss[idx] = static_cast<T>(loss_val);

    // if (idx < 10) {
    //     printf("[DEBUG][Metax] idx=%d target=%d prob=%.6f loss=%.6f\n", idx, tgt, prob, loss_val);
    // }
}
#endif 