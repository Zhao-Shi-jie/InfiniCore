#ifndef __INFINIOP_SPARSE_SCATTER_API_H__
#define __INFINIOP_SPARSE_SCATTER_API_H__

#include "../operator_descriptor.h"
#include "../spvec_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSparseScatterDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSparseScatterDescriptor(
    infiniopHandle_t handle,
    infiniopSparseScatterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopSpVecDescriptor_t input_desc);

__INFINI_C __export infiniStatus_t infiniopGetSparseScatterWorkspaceSize(infiniopSparseScatterDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSparseScatter(
    infiniopSparseScatterDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySparseScatterDescriptor(infiniopSparseScatterDescriptor_t desc);

#endif // __INFINIOP_SPARSE_SCATTER_API_H__
