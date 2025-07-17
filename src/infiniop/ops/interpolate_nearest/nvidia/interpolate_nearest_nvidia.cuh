#ifndef __INTERPOLATE_NEAREST_NVIDIA_CUH__
#define __INTERPOLATE_NEAREST_NVIDIA_CUH__

#include "../../../devices/nvidia/nvidia_handle.h"
#include "../interpolate_nearest.h"

DESCRIPTOR(nvidia)

namespace op::interpolate_nearest::nvidia {

template <typename T>
__global__ void interpolate_nearest_kernel(T *output, const T *input,
                                           int batch_size, int channels,
                                           int input_height, int input_width,
                                           int output_height, int output_width,
                                           float scale_h, float scale_w) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * output_height * output_width;

  if (idx < total_elements) {
    int temp = idx;
    int w = temp % output_width;
    temp /= output_width;
    int h = temp % output_height;
    temp /= output_height;
    int c = temp % channels;
    int b = temp / channels;

    // Calculate nearest input coordinates
    int input_h = min((int)(h / scale_h), input_height - 1);
    int input_w = min((int)(w / scale_w), input_width - 1);

    // Calculate input index
    int input_idx =
        ((b * channels + c) * input_height + input_h) * input_width + input_w;

    output[idx] = input[input_idx];
  }
}

} // namespace op::interpolate_nearest::nvidia

#endif // __INTERPOLATE_NEAREST_NVIDIA_CUH__
