#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/spmv.h"

#ifdef ENABLE_CPU_API
#include "cpu/spmv_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/spmv_nvidia.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/spmv_bang.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/spmv_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t x_desc,
    void *y,
    void const *x) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::spmv::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::spmv::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                         \
            a_desc,                                                         \
            x_desc,                                                         \
            y,                                                              \
            x)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetSpMVWorkspaceSize(infiniopSpMVDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<const op::spmv::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    void const *x,
    float alpha,
    float beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::spmv::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, x, alpha, beta, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroySpMVDescriptor(infiniopSpMVDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::spmv::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
