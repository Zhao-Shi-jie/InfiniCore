#ifndef __SPMV_CUDA_H__
#define __SPMV_CUDA_H__

#include "../../../../../include/infiniop.h"

namespace op::spmv::cuda {

// CUDA版本的CSR格式SpMV实现
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
    void *stream);

} // namespace op::spmv::cuda

#endif // __SPMV_CUDA_H__
