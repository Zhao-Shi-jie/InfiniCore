#ifndef __INFINIOP_SDDMM_API_H__
#define __INFINIOP_SDDMM_API_H__

#include "../operator_descriptor.h"
#include "../spmat_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSDDMMDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSDDMMDescriptor(
    infiniopHandle_t handle,
    infiniopSDDMMDescriptor_t *desc_ptr,
    infiniopSpMatDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc);

__INFINI_C __export infiniStatus_t infiniopGetSDDMMWorkspaceSize(infiniopSDDMMDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSDDMM(
    infiniopSDDMMDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c_values,
    void const *a,
    void const *b,
    float alpha,
    float beta,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySDDMMDescriptor(infiniopSDDMMDescriptor_t desc);

#endif // __INFINIOP_SDDMM_API_H__
