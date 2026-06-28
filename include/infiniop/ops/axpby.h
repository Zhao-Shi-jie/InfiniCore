#ifndef __INFINIOP_AXPBY_API_H__
#define __INFINIOP_AXPBY_API_H__

#include "../operator_descriptor.h"
#include "../spvec_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAxpbyDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAxpbyDescriptor(
    infiniopHandle_t handle,
    infiniopAxpbyDescriptor_t *desc_ptr,
    infiniopSpVecDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc);

__INFINI_C __export infiniStatus_t infiniopGetAxpbyWorkspaceSize(infiniopAxpbyDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAxpby(
    infiniopAxpbyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    float alpha,
    float beta,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAxpbyDescriptor(infiniopAxpbyDescriptor_t desc);

#endif // __INFINIOP_AXPBY_API_H__
