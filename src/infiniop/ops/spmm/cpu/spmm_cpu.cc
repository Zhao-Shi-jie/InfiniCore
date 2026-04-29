#include "spmm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::spmm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t values_desc,
    size_t rows,
    size_t cols) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    auto result = SpmmInfo::create(c_desc, b_desc, values_desc, rows, cols);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
static void csr_spmm(
    T *C,
    const int32_t *row_offsets,
    const int32_t *col_indices,
    const T *values,
    const T *B,
    size_t m, size_t n,
    float alpha, float beta) {

    if (beta == 0.0f) {
        std::memset(C, 0, m * n * sizeof(T));
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < m * n; ++i) {
            C[i] = static_cast<T>(static_cast<float>(C[i]) * beta);
        }
    }

    for (size_t row = 0; row < m; ++row) {
        int32_t row_start = row_offsets[row];
        int32_t row_end = row_offsets[row + 1];
        for (int32_t idx = row_start; idx < row_end; ++idx) {
            int32_t col = col_indices[idx];
            float val = alpha * static_cast<float>(values[idx]);
            for (size_t j = 0; j < n; ++j) {
                C[row * n + j] = static_cast<T>(
                    static_cast<float>(C[row * n + j]) +
                    val * static_cast<float>(B[col * n + j]));
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void * /*workspace*/,
    size_t /*workspace_size*/,
    void *c,
    const void *row_offsets,
    const void *col_indices,
    const void *values,
    const void *b,
    float alpha,
    float beta,
    void * /*stream*/) const {

    const auto *row_off = reinterpret_cast<const int32_t *>(row_offsets);
    const auto *col_idx = reinterpret_cast<const int32_t *>(col_indices);

    switch (_dtype) {
    case INFINI_DTYPE_F32:
        csr_spmm(
            reinterpret_cast<float *>(c),
            row_off, col_idx,
            reinterpret_cast<const float *>(values),
            reinterpret_cast<const float *>(b),
            _info.m, _info.n, alpha, beta);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F64:
        csr_spmm(
            reinterpret_cast<double *>(c),
            row_off, col_idx,
            reinterpret_cast<const double *>(values),
            reinterpret_cast<const double *>(b),
            _info.m, _info.n, alpha, beta);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::spmm::cpu
