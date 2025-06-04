#include "spmv_cpu.h"
#include "../info.h"
#include <cstring>

#ifdef ENABLE_OMP
#include <omp.h>
#endif

namespace op::spmv::cpu {

// CSR格式的SpMV实现模板
template <typename T>
static void spmv_csr_impl(
    T *y, const T *x, const T *values,
    const int32_t *row_ptr, const int32_t *col_idx,
    size_t num_rows) {

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < num_rows; ++i) {
        T sum = 0;
        for (int32_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

infiniStatus_t spmv_csr(
    infiniopHandle_t handle,
    void *y,
    const void *x,
    const void *values,
    const void *row_indices,
    const void *col_indices,
    size_t num_rows,
    size_t num_cols,
    size_t nnz,
    infiniDtype_t dtype,
    void *stream) {

    // 参数验证
    auto validation_result = validateSpMVCSR(
        y, x, values, row_indices, col_indices,
        num_rows, num_cols, nnz, dtype);
    CHECK_OR_RETURN(validation_result == INFINI_STATUS_SUCCESS, validation_result);

    // 根据数据类型选择对应的实现
    switch (dtype) {
    case INFINI_DTYPE_F32:
        spmv_csr_impl(
            static_cast<float *>(y),
            static_cast<const float *>(x),
            static_cast<const float *>(values),
            static_cast<const int32_t *>(row_indices),
            static_cast<const int32_t *>(col_indices),
            num_rows);
        break;

    case INFINI_DTYPE_F64:
        spmv_csr_impl(
            static_cast<double *>(y),
            static_cast<const double *>(x),
            static_cast<const double *>(values),
            static_cast<const int32_t *>(row_indices),
            static_cast<const int32_t *>(col_indices),
            num_rows);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv::cpu
