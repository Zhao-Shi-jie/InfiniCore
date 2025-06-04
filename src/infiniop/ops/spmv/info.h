#ifndef __SPMV_INFO_H__
#define __SPMV_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"

namespace op::spmv {

// 验证SpMV CSR操作的参数是否合法
inline infiniStatus_t validateSpMVCSR(
    const void *y, const void *x, const void *values,
    const void *row_indices, const void *col_indices,
    size_t num_rows, size_t num_cols, size_t nnz,
    infiniDtype_t dtype) {

    // 检查指针是否为空
    CHECK_OR_RETURN(y && x && values && row_indices && col_indices,
                    INFINI_STATUS_BAD_PARAM);

    // 检查数据类型是否支持
    CHECK_OR_RETURN(dtype == INFINI_DTYPE_F32 || dtype == INFINI_DTYPE_F64 || dtype == INFINI_DTYPE_F16, INFINI_STATUS_BAD_TENSOR_DTYPE);

    // 检查矩阵维度
    CHECK_OR_RETURN(num_rows > 0 && num_cols > 0 && nnz > 0,
                    INFINI_STATUS_BAD_PARAM);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv

#endif // __SPMV_INFO_H__