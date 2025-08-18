#ifndef __GRAD_CUDA_H__
#define __GRAD_CUDA_H__

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

// CUDA kernel 实现
__global__ void compute_bias_grad_kernel(const void *grad_output,
                                         void *grad_bias, int batch_size,
                                         int channels, int spatial_size,
                                         cudnnDataType_t data_type) {

  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;

  if (data_type == CUDNN_DATA_BFLOAT16) {
    const __nv_bfloat16 *grad_out =
        reinterpret_cast<const __nv_bfloat16 *>(grad_output);
    __nv_bfloat16 *grad_b = reinterpret_cast<__nv_bfloat16 *>(grad_bias);

    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
      for (int s = 0; s < spatial_size; s++) {
        int idx = n * channels * spatial_size + c * spatial_size + s;
        sum += __bfloat162float(grad_out[idx]);
      }
    }
    grad_b[c] = __float2bfloat16(sum);

  } else if (data_type == CUDNN_DATA_HALF) {
    const __half *grad_out = reinterpret_cast<const __half *>(grad_output);
    __half *grad_b = reinterpret_cast<__half *>(grad_bias);

    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
      for (int s = 0; s < spatial_size; s++) {
        int idx = n * channels * spatial_size + c * spatial_size + s;
        sum += __half2float(grad_out[idx]);
      }
    }
    grad_b[c] = __float2half(sum);

  } else if (data_type == CUDNN_DATA_FLOAT) {
    const float *grad_out = reinterpret_cast<const float *>(grad_output);
    float *grad_b = reinterpret_cast<float *>(grad_bias);

    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
      for (int s = 0; s < spatial_size; s++) {
        int idx = n * channels * spatial_size + c * spatial_size + s;
        sum += grad_out[idx];
      }
    }
    grad_b[c] = sum;
  }
}

// 启动函数声明
infiniStatus_t launch_bias_grad_kernel(const void *grad_output, void *grad_bias,
                                       const int *grad_output_dims,
                                       size_t conv_ndim,
                                       cudnnDataType_t data_type,
                                       cudaStream_t stream);
#endif // __GRAD_CUDA_H__
