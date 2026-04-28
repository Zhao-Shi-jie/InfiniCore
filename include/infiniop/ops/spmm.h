#ifndef __INFINIOP_SPMM_API_H__
#define __INFINIOP_SPMM_API_H__

#include "../operator_descriptor.h"
#include "../spmat_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSpMMDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSpMMDescriptor(
    infiniopHandle_t handle,
    infiniopSpMMDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc);

__INFINI_C __export infiniStatus_t infiniopGetSpMMWorkspaceSize(infiniopSpMMDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSpMM(
    infiniopSpMMDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    void const *b,
    float alpha,
    float beta,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySpMMDescriptor(infiniopSpMMDescriptor_t desc);

#endif // __INFINIOP_SPMM_API_H__
