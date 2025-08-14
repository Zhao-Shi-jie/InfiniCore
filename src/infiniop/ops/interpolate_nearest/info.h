#ifndef __INTERPOLATE_NEAREST_INFO_H__
#define __INTERPOLATE_NEAREST_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <cstddef>

struct InterpolateNearestInfo {
    size_t batch_size;
    size_t channels;
    size_t input_height;
    size_t input_width;
    size_t output_height;
    size_t output_width;
    float scale_h;
    float scale_w;
    infiniDtype_t dtype;

    size_t input_stride[4];
    size_t output_stride[4];

    static infiniStatus_t create(
        InterpolateNearestInfo *info,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc) {

        // Check that input and output have same dtype
        if (input_desc->dtype() != output_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Check that both tensors are 4D (NCHW format)
        if (input_desc->ndim() != 4 || output_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto input_shape = input_desc->shape();
        auto output_shape = output_desc->shape();

        // Check that batch size and channels match
        if (input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        info->batch_size = input_shape[0];
        info->channels = input_shape[1];
        info->input_height = input_shape[2];
        info->input_width = input_shape[3];
        info->output_height = output_shape[2];
        info->output_width = output_shape[3];

        info->scale_h = (float)info->output_height / info->input_height;
        info->scale_w = (float)info->output_width / info->input_width;

        info->dtype = input_desc->dtype();

        auto input_stride = input_desc->strides();
        auto output_stride = output_desc->strides();
        for (int i = 0; i < 4; ++i) {
            info->input_stride[i] = input_stride[i];
            info->output_stride[i] = output_stride[i];
        }

        return INFINI_STATUS_SUCCESS;
    }
};

#endif
