#include "spmm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::spmm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SpMMInfo::create(c_desc, a_desc, b_desc);
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
    const SpMMInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *c,
    const void *b,
    float alpha,
    float beta) {
    auto values = reinterpret_cast<const Tdata *>(a_desc->values());
    auto crow_indices = reinterpret_cast<const Tindex *>(a_desc->crowIndices());
    auto col_indices = reinterpret_cast<const Tindex *>(a_desc->colIndices());
    auto b_data = reinterpret_cast<const Tdata *>(b);
    auto c_data = reinterpret_cast<Tdata *>(c);

#pragma omp parallel for
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(info.m); ++row) {
        for (size_t col = 0; col < info.n; ++col) {
            auto c_offset = row * info.c_matrix.row_stride + col * info.c_matrix.col_stride;
            float acc = 0;
            for (Tindex ptr = crow_indices[row]; ptr < crow_indices[row + 1]; ++ptr) {
                auto k = static_cast<size_t>(col_indices[ptr]);
                auto b_offset = k * info.b_matrix.row_stride + col * info.b_matrix.col_stride;
                acc += utils::cast<float>(values[ptr]) * utils::cast<float>(b_data[b_offset]);
            }
            if (beta == 0) {
                c_data[c_offset] = utils::cast<Tdata>(alpha * acc);
            } else {
                c_data[c_offset] = utils::cast<Tdata>(alpha * acc + beta * utils::cast<float>(c_data[c_offset]));
            }
        }
    }
}

template <typename Tdata>
infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const SpMMInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *c,
    const void *b,
    float alpha,
    float beta) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        calculate<Tdata, int32_t>(info, a_desc, c, b, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I64:
        calculate<Tdata, int64_t>(info, a_desc, c, b, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<fp16_t>(_index_dtype, _info, _a_desc, c, b, alpha, beta);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<bf16_t>(_index_dtype, _info, _a_desc, c, b, alpha, beta);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_index_dtype, _info, _a_desc, c, b, alpha, beta);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::spmm::cpu
