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

template <typename T_in, typename T_out>
__global__ void softmaxCrossEntropy_generic(
    T_out* loss,               // 输出：逐元素损失（与target同形状）
    const T_in* logits,
    const int* target,
    int N, int C, long long inner_size) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_elements = N * inner_size;
    if (idx >= total_elements) return;  // 严格越界检查

    int n = idx / inner_size;
    int inner_idx = idx % inner_size;
    int tgt = target[idx];

    // 处理PyTorch默认忽略索引（-100）
    const int ignore_index = -100;
    if (tgt == ignore_index) {
        loss[idx] = static_cast<T_out>(0.0f);  // 忽略的位置损失设为0
        return;
    }

    // 处理无效类别（超出范围）
    if (tgt < 0 || tgt >= C) {
        loss[idx] = static_cast<T_out>(1e9f);  // 错误标记
        return;
    }

    // 计算Softmax（数值稳定版）
    float max_val = -CUDART_INF_F;
    for (int c = 0; c < C; ++c) {
        long long offset = ((long long)n * C + c) * inner_size + inner_idx;
        max_val = fmaxf(max_val, static_cast<float>(logits[offset]));
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        long long offset = ((long long)n * C + c) * inner_size + inner_idx;
        sum_exp += expf(static_cast<float>(logits[offset]) - max_val);
    }

    // 计算交叉熵
    long long tgt_offset = ((long long)n * C + tgt) * inner_size + inner_idx;
    float logit_tgt = static_cast<float>(logits[tgt_offset]);
    float prob = expf(logit_tgt - max_val) / sum_exp;
    loss[idx] = static_cast<T_out>(-logf(prob));
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

    const auto &original_logits_shape = logits_desc->shape();
    auto opaque = new Opaque(handle->internal());
    
    // 处理一维情况：将 (C,) 扩展为 (1, C)
    if (original_logits_shape.size() == 1) {
        opaque->logits_shape = {1, original_logits_shape[0]};  // (C,) -> (1, C)
    } else {
        opaque->logits_shape = original_logits_shape;
    }

    // 使用扩展后的形状计算工作空间
    const auto &logits_shape = opaque->logits_shape;
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
    // 使用扩展后的形状（已经处理了一维->二维的转换）
    const auto &shape = _opaque->logits_shape;
    
    int N = shape[0];
    int C = shape[1];
    long long inner_size = 1;
    for (size_t i = 2; i < shape.size(); ++i)
        inner_size *= shape[i];
    long long total_elements = N * inner_size;

    // 第一步：调用核函数计算逐元素损失
    dim3 blockSize(256);
    dim3 gridSize((total_elements + blockSize.x - 1) / blockSize.x);

    // 根据数据类型调用核函数
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

    // 第二步：计算均值（代码保持不变）
    std::vector<int> h_target(total_elements);
    cudaMemcpyAsync(h_target.data(), target, total_elements * sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)stream);

    float total_sum = 0.0f;
    int valid_count = 0;
    const int ignore_index = -100;

    if (_dtype == INFINI_DTYPE_F32) {
        std::vector<float> h_loss(total_elements);
        cudaMemcpyAsync(h_loss.data(), loss, total_elements * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
        cudaStreamSynchronize((cudaStream_t)stream);
        
        for (long long i = 0; i < total_elements; ++i) {
            if (h_target[i] != ignore_index) {
                total_sum += h_loss[i];
                valid_count++;
            }
        }
        
        float mean_loss = (valid_count > 0) ? (total_sum / valid_count) : 0.0f;
        cudaMemcpyAsync(loss, &mean_loss, sizeof(float), cudaMemcpyHostToDevice, (cudaStream_t)stream);
        
    } else if (_dtype == INFINI_DTYPE_F16) {
        std::vector<half> h_loss(total_elements);
        cudaMemcpyAsync(h_loss.data(), loss, total_elements * sizeof(half), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
        cudaStreamSynchronize((cudaStream_t)stream);
        
        for (long long i = 0; i < total_elements; ++i) {
            if (h_target[i] != ignore_index) {
                total_sum += __half2float(h_loss[i]);
                valid_count++;
            }
        }
        
        float mean_loss_f = (valid_count > 0) ? (total_sum / valid_count) : 0.0f;
        half mean_loss = __float2half(mean_loss_f);
        cudaMemcpyAsync(loss, &mean_loss, sizeof(half), cudaMemcpyHostToDevice, (cudaStream_t)stream);
        
    } else if (_dtype == INFINI_DTYPE_BF16) {
        std::vector<__nv_bfloat16> h_loss(total_elements);
        cudaMemcpyAsync(h_loss.data(), loss, total_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
        cudaStreamSynchronize((cudaStream_t)stream);
        
        for (long long i = 0; i < total_elements; ++i) {
            if (h_target[i] != ignore_index) {
                total_sum += __bfloat162float(h_loss[i]);
                valid_count++;
            }
        }
        
        float mean_loss_f = (valid_count > 0) ? (total_sum / valid_count) : 0.0f;
        __nv_bfloat16 mean_loss = __float2bfloat16(mean_loss_f);
        cudaMemcpyAsync(loss, &mean_loss, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, (cudaStream_t)stream);
    }

    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}
} // namespace op::cross_entropy_loss::nvidia
