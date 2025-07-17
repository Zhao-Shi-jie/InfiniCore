#include "../../tensor.h"
#include "interpolate_nearest.h"

infiniStatus_t InterpolateNearestInfo::create(
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

    return INFINI_STATUS_SUCCESS;
}
