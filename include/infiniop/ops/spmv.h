#ifndef __INFINIOP_SPMV_API_H__
#define __INFINIOP_SPMV_API_H__

#include "../operator_descriptor.h"
#include "../spmat_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSpMVDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t x_desc,
    void *y,
    void const *x);

__INFINI_C __export infiniStatus_t infiniopGetSpMVWorkspaceSize(infiniopSpMVDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    void const *x,
    float alpha,
    float beta,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySpMVDescriptor(infiniopSpMVDescriptor_t desc);

#endif // __INFINIOP_SPMV_API_H__
