#ifndef __INFINIOP_SPARSE_GATHER_API_H__
#define __INFINIOP_SPARSE_GATHER_API_H__

#include "../operator_descriptor.h"
#include "../spvec_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSparseGatherDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSparseGatherDescriptor(
    infiniopHandle_t handle,
    infiniopSparseGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopSpVecDescriptor_t pattern_desc,
    infiniopTensorDescriptor_t input_desc,
    void *output,
    const void *input);

__INFINI_C __export infiniStatus_t infiniopGetSparseGatherWorkspaceSize(infiniopSparseGatherDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSparseGather(
    infiniopSparseGatherDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySparseGatherDescriptor(infiniopSparseGatherDescriptor_t desc);

#endif // __INFINIOP_SPARSE_GATHER_API_H__
