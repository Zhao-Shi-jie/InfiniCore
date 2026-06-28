#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/axpby.h"

#ifdef ENABLE_CPU_API
#include "cpu/axpby_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/axpby_nvidia.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/axpby_bang.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/axpby_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateAxpbyDescriptor(
    infiniopHandle_t handle,
    infiniopAxpbyDescriptor_t *desc_ptr,
    infiniopSpVecDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

#define CREATE(CASE, NAMESPACE)                                              \
    case CASE:                                                               \
        return op::axpby::NAMESPACE::Descriptor::create(                     \
            handle,                                                          \
            reinterpret_cast<op::axpby::NAMESPACE::Descriptor **>(desc_ptr), \
            x_desc,                                                          \
            y_desc)

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

__INFINI_C infiniStatus_t infiniopGetAxpbyWorkspaceSize(infiniopAxpbyDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<const op::axpby::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopAxpby(
    infiniopAxpbyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    float alpha,
    float beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<const op::axpby::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, alpha, beta, stream)

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

__INFINI_C infiniStatus_t infiniopDestroyAxpbyDescriptor(infiniopAxpbyDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        delete reinterpret_cast<const op::axpby::NAMESPACE::Descriptor *>(desc); \
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
