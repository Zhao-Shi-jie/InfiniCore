#ifndef __INFINIOP_SPMAT_DESCRIPTOR_API_H__
#define __INFINIOP_SPMAT_DESCRIPTOR_API_H__

#include "../infinicore.h"
#include "tensor_descriptor.h"

typedef enum {
    INFINIOP_SPMAT_FORMAT_CSR = 0,
    INFINIOP_SPMAT_FORMAT_ELL = 1,
    INFINIOP_SPMAT_FORMAT_SELL = 2,
    INFINIOP_SPMAT_FORMAT_SELL_SIGMA_C = 3,
    INFINIOP_SPMAT_FORMAT_COO = 4,
} infiniopSpMatFormat_t;

struct InfiniopSpMatDescriptor;

typedef struct InfiniopSpMatDescriptor *infiniopSpMatDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateCsrSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t crow_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *crow_indices,
    void const *col_indices);

__INFINI_C __export infiniStatus_t infiniopCreateCooSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *row_indices,
    void const *col_indices);

__INFINI_C __export infiniStatus_t infiniopCreateEllSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t ell_width,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *col_indices);

__INFINI_C __export infiniStatus_t infiniopCreateSellSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t slice_height,
    size_t num_slices,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t slice_offsets_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *slice_offsets,
    void const *col_indices);

__INFINI_C __export infiniStatus_t infiniopCreateSellSigmaCSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t slice_height,
    size_t sigma,
    size_t num_slices,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t slice_offsets_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    void const *values,
    void const *slice_offsets,
    void const *col_indices,
    void const *row_indices);

__INFINI_C __export infiniStatus_t infiniopDestroySpMatDescriptor(infiniopSpMatDescriptor_t desc);

#endif // __INFINIOP_SPMAT_DESCRIPTOR_API_H__
