#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdio.h>
#include <numeric>
#include <vector>
#include <memory>

#include "cross_entropy_loss_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"

namespace op::cross_entropy_loss::nvidia {
namespace cuda {

// 通用 CrossEntropyLoss kernel：支持 logits shape = (N, C, D1, ..., Dn)，row-major
// 显式展开 idx 为 (n, d1, ..., dn)，并在 dim=1 上执行 softmax

template <typename T_in, typename T_out>
__global__ void softmaxCrossEntropy_generic(
    T_out* loss,
    const T_in* logits,
    const int* target,
    int N, int C, long long inner_size) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_elements = N * inner_size;
    if (idx >= total_elements) return;

    int n = idx / inner_size;
    int inner_idx = idx % inner_size; // index within spatial dims

    int tgt = target[idx];
    if (tgt < 0 || tgt >= C) {
        loss[idx] = static_cast<T_out>(1e9f);
        return;
    }

    // offset for (n, c, ...), base index is (n * C + c) * inner_size + inner_idx
    float max_val = -CUDART_INF_F;
    for (int c = 0; c < C; ++c) {
        long long offset = ((long long)n * C + c) * inner_size + inner_idx;
        float val = static_cast<float>(logits[offset]);
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.f;
    for (int c = 0; c < C; ++c) {
        long long offset = ((long long)n * C + c) * inner_size + inner_idx;
        sum_exp += expf(static_cast<float>(logits[offset]) - max_val);
    }

    long long tgt_offset = ((long long)n * C + tgt) * inner_size + inner_idx;
    float logit_tgt = static_cast<float>(logits[tgt_offset]);
    float prob = expf(logit_tgt - max_val) / sum_exp;
    float ce = -logf(prob);

    loss[idx] = static_cast<T_out>(ce);

    // if (idx < 10) {
    //     printf("[DEBUG] idx=%lld target=%d prob=%.6f loss=%.6f\n", idx, tgt, prob, ce);
    // }
}

} // namespace cuda

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    std::vector<size_t> logits_shape;
    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_ptr)
        : internal(internal_ptr) {}
    ~Opaque() = default;
};

Descriptor::~Descriptor() {
    if (_opaque) delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t loss_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t target_desc) {

#ifdef ENABLE_NVIDIA_API
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = logits_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    const auto &logits_shape = logits_desc->shape();
    auto opaque = new Opaque(handle->internal());
    opaque->logits_shape = logits_shape;

    long long total_elements = logits_shape[0];
    for (size_t i = 2; i < logits_shape.size(); ++i)
        total_elements *= logits_shape[i];

    size_t workspace_size = total_elements * logits_shape[1] * sizeof(float);
    *desc_ptr = new Descriptor(dtype, workspace_size, opaque, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *loss,
    const void *logits, const void *target, void *stream) const {

#ifdef ENABLE_NVIDIA_API
    const auto &shape = _opaque->logits_shape;
    if (shape.size() < 2) return INFINI_STATUS_NOT_SUPPORTED;

    int N = shape[0];
    int C = shape[1];
    long long inner_size = 1;
    for (size_t i = 2; i < shape.size(); ++i)
        inner_size *= shape[i];

    long long total_elements = N * inner_size;
    dim3 blockSize(256);
    dim3 gridSize((total_elements + blockSize.x - 1) / blockSize.x);

    if (_dtype == INFINI_DTYPE_F32) {
        cuda::softmaxCrossEntropy_generic<float, float><<<gridSize, blockSize, 0, (cudaStream_t)stream>>>(
            (float*)loss, (const float*)logits, (const int*)target,
            N, C, inner_size);
    } else if (_dtype == INFINI_DTYPE_F16) {
        cuda::softmaxCrossEntropy_generic<half, half><<<gridSize, blockSize, 0, (cudaStream_t)stream>>>(
            (half*)loss, (const half*)logits, (const int*)target,
            N, C, inner_size);
    } else if (_dtype == INFINI_DTYPE_BF16) {
        cuda::softmaxCrossEntropy_generic<__nv_bfloat16, __nv_bfloat16><<<gridSize, blockSize, 0, (cudaStream_t)stream>>>(
            (__nv_bfloat16*)loss, (const __nv_bfloat16*)logits, (const int*)target,
            N, C, inner_size);
    }

    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

} // namespace op::cross_entropy_loss::nvidia