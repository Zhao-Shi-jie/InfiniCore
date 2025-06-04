#ifndef __INFINIOP_SPMV_API_H__
#define __INFINIOP_SPMV_API_H__

#include "../operator_descriptor.h"

// 稀疏矩阵向量乘法 - CSR格式
// y = A * x, 其中A是CSR格式的稀疏矩阵
__C __export infiniStatus_t infiniopSpMV_csr(
    infiniopHandle_t handle,
    void *y,                 // 输出向量，大小为 num_rows
    const void *x,           // 输入向量，大小为 num_cols
    const void *values,      // 非零元素值数组，大小为 nnz
    const void *row_indices, // 行指针数组，大小为 num_rows+1
    const void *col_indices, // 列索引数组，大小为 nnz
    size_t num_rows,         // 矩阵行数
    size_t num_cols,         // 矩阵列数
    size_t nnz,              // 非零元素个数
    infiniDtype_t dtype,     // 数据类型
    void *stream);           // 计算流

#endif
