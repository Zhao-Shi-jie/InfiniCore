#include "spmv_cpu.h"

namespace op::spmv::cpu {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    SparseFormat format) {

    auto info_result = SpMVInfo::create(y_desc, x_desc, values_desc, row_indices_desc, col_indices_desc, format);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *values,
    const void *row_indices,
    const void *col_indices,
    void *stream) const {

    // Placeholder for future CPU implementation
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

} // namespace op::spmv::cpu
