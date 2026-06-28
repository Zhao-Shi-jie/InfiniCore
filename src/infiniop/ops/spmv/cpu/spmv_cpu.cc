#include "spmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::spmv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t x_desc,
    void *y,
    const void *x) {
    (void)y;
    (void)x;
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SpMVInfo::create(y_desc, a_desc, x_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        a_desc->crowIndicesDesc()->dtype(),
        result.take(),
        a_desc,
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
void calculate(
    const SpMVInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *y,
    const void *x,
    float alpha,
    float beta) {
    auto values = reinterpret_cast<const Tdata *>(a_desc->values());
    auto crow_indices = reinterpret_cast<const Tindex *>(a_desc->crowIndices());
    auto col_indices = reinterpret_cast<const Tindex *>(a_desc->colIndices());
    auto x_data = reinterpret_cast<const Tdata *>(x);
    auto y_data = reinterpret_cast<Tdata *>(y);

#pragma omp parallel for
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(info.m); ++row) {
        float acc = 0;
        for (Tindex ptr = crow_indices[row]; ptr < crow_indices[row + 1]; ++ptr) {
            auto k = static_cast<size_t>(col_indices[ptr]);
            acc += utils::cast<float>(values[ptr]) * utils::cast<float>(x_data[k * info.x_vector.stride]);
        }

        auto y_offset = row * info.y_vector.stride;
        if (beta == 0) {
            y_data[y_offset] = utils::cast<Tdata>(alpha * acc);
        } else {
            y_data[y_offset] = utils::cast<Tdata>(alpha * acc + beta * utils::cast<float>(y_data[y_offset]));
        }
    }
}

template <typename Tdata>
infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const SpMVInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *y,
    const void *x,
    float alpha,
    float beta) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        calculate<Tdata, int32_t>(info, a_desc, y, x, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I64:
        calculate<Tdata, int64_t>(info, a_desc, y, x, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<fp16_t>(_index_dtype, _info, _a_desc, y, x, alpha, beta);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<bf16_t>(_index_dtype, _info, _a_desc, y, x, alpha, beta);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_index_dtype, _info, _a_desc, y, x, alpha, beta);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::spmv::cpu
