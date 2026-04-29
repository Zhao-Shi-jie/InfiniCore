#ifndef __INFINIOP_SPMM_API_H__
#define __INFINIOP_SPMM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSpmmDescriptor_t;

/**
 * Create a SpMM descriptor.
 *
 * @param handle        infiniop handle
 * @param desc_ptr      output descriptor pointer
 * @param c_desc        dense output tensor descriptor, shape [m, n]
 * @param b_desc        dense input tensor descriptor, shape [k, n]
 * @param values_desc   CSR non-zero values descriptor, shape [nnz], dtype T
 * @param rows          number of rows of the sparse matrix (m)
 * @param cols          number of columns of the sparse matrix (k)
 */
__INFINI_C __export infiniStatus_t infiniopCreateSpmmDescriptor(infiniopHandle_t handle,
                                                                infiniopSpmmDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t c_desc,
                                                                infiniopTensorDescriptor_t b_desc,
                                                                infiniopTensorDescriptor_t values_desc,
                                                                size_t rows,
                                                                size_t cols);

__INFINI_C __export infiniStatus_t infiniopGetSpmmWorkspaceSize(infiniopSpmmDescriptor_t desc, size_t *size);

/**
 * Execute SpMM: C = alpha * CSR(A) * B + beta * C.
 *
 * @param desc            SpMM descriptor
 * @param workspace       temporary workspace buffer
 * @param workspace_size  size of workspace in bytes
 * @param c               dense output data pointer  [m, n]
 * @param row_offsets     CSR row-pointer array      [m+1], I32
 * @param col_indices     CSR column-index array     [nnz], I32
 * @param values          CSR non-zero values        [nnz], T
 * @param b               dense input data pointer   [k, n]
 * @param alpha           scalar multiplier for A*B
 * @param beta            scalar multiplier for C
 * @param stream          device stream (null for CPU)
 */
__INFINI_C __export infiniStatus_t infiniopSpmm(infiniopSpmmDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *c,
                                                const void *row_offsets,
                                                const void *col_indices,
                                                const void *values,
                                                const void *b,
                                                float alpha,
                                                float beta,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySpmmDescriptor(infiniopSpmmDescriptor_t desc);

#endif // __INFINIOP_SPMM_API_H__
