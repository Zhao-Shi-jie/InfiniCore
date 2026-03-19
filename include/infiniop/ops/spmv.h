#ifndef __INFINIOP_SPMV_API_H__
#define __INFINIOP_SPMV_API_H__

#include "../operator_descriptor.h"
#include <cstddef>

typedef struct InfiniopDescriptor *infiniopSpMVDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    size_t num_cols,
    size_t num_rows,
    size_t nnz);

__INFINI_C __export infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *y,                 // 输出向量
    const void *x,           // 输入向量
    const void *values,      // 非零元素值数组
    const void *row_ptr,     // 行偏移数组
    const void *col_indices, // 列索引数组
    void *stream);           // 计算流

__INFINI_C __export infiniStatus_t infiniopDestroySpMVDescriptor(infiniopSpMVDescriptor_t desc);

#endif
