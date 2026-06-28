#ifndef __INFINIOP_SPVV_API_H__
#define __INFINIOP_SPVV_API_H__

#include "../operator_descriptor.h"
#include "../spvec_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSpVVDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSpVVDescriptor(infiniopHandle_t handle,
                                                                infiniopSpVVDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y_desc,
                                                                infiniopSpVecDescriptor_t a_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                const void *x);

__INFINI_C __export infiniStatus_t infiniopGetSpVVWorkspaceSize(infiniopSpVVDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSpVV(infiniopSpVVDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySpVVDescriptor(infiniopSpVVDescriptor_t desc);

#endif // __INFINIOP_SPVV_API_H__
