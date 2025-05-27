#ifndef __INFINIOP_SPMV_API_H__
#define __INFINIOP_SPMV_API_H__

#include "../operator_descriptor.h"

/**
 * @brief SpMV descriptor opaque handle.
 */
typedef struct infiniopDescriptor *infiniopSpMVDescriptor_t;

/**
 * @brief Create a new SpMV descriptor.
 *
 * @param handle Device handle
 * @param desc_ptr Output pointer to the created descriptor
 * @param y_desc Output tensor descriptor (vector)
 * @param x_desc Input tensor descriptor (vector)
 * @param values_desc Values tensor descriptor (non-zero elements)
 * @param row_indices_desc Row indices/pointers tensor descriptor
 * @param col_indices_desc Column indices tensor descriptor
 * @param sparse_format Format of the sparse matrix (0 for CSR, 1 for COO)
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    int sparse_format);

/**
 * @brief Get workspace size required for the SpMV operation.
 *
 * @param desc SpMV descriptor
 * @param size Output pointer to workspace size
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopGetSpMVWorkspaceSize(
    infiniopSpMVDescriptor_t desc,
    size_t *size);

/**
 * @brief Execute the SpMV operation: y = A * x
 *
 * @param desc SpMV descriptor
 * @param workspace Workspace memory
 * @param workspace_size Size of workspace memory
 * @param y Output vector
 * @param x Input vector
 * @param values Non-zero values of the sparse matrix
 * @param row_indices Row indices (CSR) or row pointers (COO)
 * @param col_indices Column indices
 * @param stream Execution stream
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *values,
    const void *row_indices,
    const void *col_indices,
    void *stream);

/**
 * @brief Destroy the SpMV descriptor.
 *
 * @param desc SpMV descriptor to destroy
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopDestroySpMVDescriptor(
    infiniopSpMVDescriptor_t desc);

#endif
