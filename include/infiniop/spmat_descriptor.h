#ifndef __INFINIOP_SPMAT_DESCRIPTOR_API_H__
#define __INFINIOP_SPMAT_DESCRIPTOR_API_H__

#include "../infinicore.h"
#include "tensor_descriptor.h"

typedef enum {
    INFINIOP_SPMAT_FORMAT_CSR = 0,
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

__INFINI_C __export infiniStatus_t infiniopDestroySpMatDescriptor(infiniopSpMatDescriptor_t desc);

#endif // __INFINIOP_SPMAT_DESCRIPTOR_API_H__
