#ifndef __INTERPOLATE_NEAREST_KERNEL_CUH__
#define __INTERPOLATE_NEAREST_KERNEL_CUH__

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

#endif // __INTERPOLATE_NEAREST_KERNEL_CUH__