#include "sddmm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::sddmm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopSpMatDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = c_desc->valuesDesc()->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

    auto result = SDDMMInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        c_desc->crowIndicesDesc()->dtype(),
        result.take(),
        c_desc,
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
static void calculateSDDMM(
    const SDDMMInfo &info,
    infiniopSpMatDescriptor_t c_desc,
    void *c_values,
    const void *a,
    const void *b,
    float alpha,
    float beta) {
    auto crow = reinterpret_cast<const Tindex *>(c_desc->crowIndices());
    auto col = reinterpret_cast<const Tindex *>(c_desc->colIndices());
    auto c = reinterpret_cast<Tdata *>(c_values);
    auto a_data = reinterpret_cast<const Tdata *>(a);
    auto b_data = reinterpret_cast<const Tdata *>(b);

#pragma omp parallel for
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(info.m); ++row) {
        for (Tindex ptr = crow[row]; ptr < crow[row + 1]; ++ptr) {
            auto c_col = static_cast<size_t>(col[ptr]);
            float acc = 0.0f;
            for (size_t k = 0; k < info.k; ++k) {
                auto a_offset = row * info.a_matrix.row_stride + k * info.a_matrix.col_stride;
                auto b_offset = k * info.b_matrix.row_stride + c_col * info.b_matrix.col_stride;
                acc += utils::cast<float>(a_data[a_offset]) * utils::cast<float>(b_data[b_offset]);
            }
            c[ptr] = utils::cast<Tdata>(alpha * acc + beta * utils::cast<float>(c[ptr]));
        }
    }
}

template <typename Tdata>
static infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const SDDMMInfo &info,
    infiniopSpMatDescriptor_t c_desc,
    void *c_values,
    const void *a,
    const void *b,
    float alpha,
    float beta) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        calculateSDDMM<Tdata, int32_t>(info, c_desc, c_values, a, b, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I64:
        calculateSDDMM<Tdata, int64_t>(info, c_desc, c_values, a, b, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c_values,
    const void *a,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<fp16_t>(_index_dtype, _info, _c_desc, c_values, a, b, alpha, beta);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<bf16_t>(_index_dtype, _info, _c_desc, c_values, a, b, alpha, beta);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_index_dtype, _info, _c_desc, c_values, a, b, alpha, beta);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::sddmm::cpu
