#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "interpolate_nearest_nvidia.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>

namespace op::interpolate_nearest::nvidia {

template <typename T>
__global__ void interpolate_nearest_kernel(T *output, const T *input,
                                           InterpolateNearestInfo info) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements =
      info.batch_size * info.channels * info.output_height * info.output_width;

  if (idx < total_elements) {
    int temp = idx;
    int w = temp % info.output_width;
    temp /= info.output_width;
    int h = temp % info.output_height;
    temp /= info.output_height;
    int c = temp % info.channels;
    int b = temp / info.channels;

    int input_h = min((size_t)floorf((float)h * (float)info.input_height /
                                     (float)info.output_height),
                      info.input_height - 1);
    int input_w = min((size_t)floorf((float)w * (float)info.input_width /
                                     (float)info.output_width),
                      info.input_width - 1);

    int input_idx = b * info.input_stride[0] + c * info.input_stride[1] +
                    input_h * info.input_stride[2] +
                    input_w * info.input_stride[3];

    int output_idx = b * info.output_stride[0] + c * info.output_stride[1] +
                     h * info.output_stride[2] + w * info.output_stride[3];

    output[output_idx] = input[input_idx];
  }
}

struct Descriptor::Opaque {
  std::shared_ptr<device::nvidia::Handle::Internal> internal;

  Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_)
      : internal(internal_) {}
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t output_desc,
                                  infiniopTensorDescriptor_t input_desc) {

  auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
  auto dtype = output_desc->dtype();

  // Check supported data types
  if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32 &&
      dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_I8) {
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
  }

  InterpolateNearestInfo info;
  CHECK_STATUS(InterpolateNearestInfo::create(&info, output_desc, input_desc));

  *desc_ptr = new Descriptor(dtype, info, 0, new Opaque{handle->internal()},
                             handle->device, handle->device_id);

  return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *output, const void *input,
                                     void *stream) const {

  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  int total_elements = _info.batch_size * _info.channels * _info.output_height *
                       _info.output_width;
  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;

  switch (_dtype) {
  case INFINI_DTYPE_F32: {
    float *typed_output = nullptr;
    const float *typed_input = nullptr;
    typed_output = reinterpret_cast<float *>(output);
    typed_input = reinterpret_cast<const float *>(input);
    interpolate_nearest_kernel<float>
        <<<grid_size, block_size, 0, cuda_stream>>>(typed_output, typed_input,
                                                    _info);
  } break;

  case INFINI_DTYPE_F16: {
    half *typed_output = nullptr;
    const half *typed_input = nullptr;
    typed_output = reinterpret_cast<half *>(output);
    typed_input = reinterpret_cast<const half *>(input);
    interpolate_nearest_kernel<half><<<grid_size, block_size, 0, cuda_stream>>>(
        typed_output, typed_input, _info);
  } break;

  case INFINI_DTYPE_BF16: {
    auto typed_output = reinterpret_cast<__nv_bfloat16 *>(output);
    auto typed_input = reinterpret_cast<const __nv_bfloat16 *>(input);
    interpolate_nearest_kernel<__nv_bfloat16>
        <<<grid_size, block_size, 0, cuda_stream>>>(typed_output, typed_input,
                                                    _info);
  } break;

  case INFINI_DTYPE_I8: {
    auto typed_output = reinterpret_cast<int8_t *>(output);
    auto typed_input = reinterpret_cast<const int8_t *>(input);
    interpolate_nearest_kernel<int8_t>
        <<<grid_size, block_size, 0, cuda_stream>>>(typed_output, typed_input,
                                                    _info);
  } break;
  default:
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
  }

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
  return INFINI_STATUS_SUCCESS;
}

} // namespace op::interpolate_nearest::nvidia
