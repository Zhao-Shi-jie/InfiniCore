#include "../../../devices/nvidia/common_nvidia.cuh"
#include "interpolate_nearest_nvidia.cuh"

namespace op::interpolate_nearest::nvidia {

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
  if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
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
  case INFINI_DTYPE_F32:
    interpolate_nearest_kernel<float>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(input), _info.batch_size,
            _info.channels, _info.input_height, _info.input_width,
            _info.output_height, _info.output_width, _info.scale_h,
            _info.scale_w);
    break;
  case INFINI_DTYPE_F16:
    interpolate_nearest_kernel<half><<<grid_size, block_size, 0, cuda_stream>>>(
        reinterpret_cast<half *>(output), reinterpret_cast<const half *>(input),
        _info.batch_size, _info.channels, _info.input_height, _info.input_width,
        _info.output_height, _info.output_width, _info.scale_h, _info.scale_w);
    break;
  default:
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
  }

  CHECK_CUDA(cudaGetLastError());
  return INFINI_STATUS_SUCCESS;
}

} // namespace op::interpolate_nearest::nvidia
