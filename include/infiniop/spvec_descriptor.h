#ifndef __INFINIOP_SPVEC_DESCRIPTOR_API_H__
#define __INFINIOP_SPVEC_DESCRIPTOR_API_H__

#include "../infinicore.h"
#include "tensor_descriptor.h"

struct InfiniopSpVecDescriptor;

typedef struct InfiniopSpVecDescriptor *infiniopSpVecDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSpVecDescriptor(
    infiniopSpVecDescriptor_t *desc_ptr,
    size_t size,
    size_t nnz,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t indices_desc,
    void const *values,
    void const *indices);

__INFINI_C __export infiniStatus_t infiniopDestroySpVecDescriptor(infiniopSpVecDescriptor_t desc);

#endif // __INFINIOP_SPVEC_DESCRIPTOR_API_H__
