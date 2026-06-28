#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/sddmm.h"

#ifdef ENABLE_CPU_API
#include "cpu/sddmm_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/sddmm_nvidia.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/sddmm_bang.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/sddmm_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateSDDMMDescriptor(
    infiniopHandle_t handle,
    infiniopSDDMMDescriptor_t *desc_ptr,
    infiniopSpMatDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
#define CREATE(CASE, NAMESPACE)                                              \
    case CASE:                                                               \
        return op::sddmm::NAMESPACE::Descriptor::create(                     \
            handle,                                                          \
            reinterpret_cast<op::sddmm::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                          \
            a_desc,                                                          \
            b_desc)

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

__INFINI_C infiniStatus_t infiniopGetSDDMMWorkspaceSize(infiniopSDDMMDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<const op::sddmm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopSDDMM(
    infiniopSDDMMDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c_values,
    void const *a,
    void const *b,
    float alpha,
    float beta,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<const op::sddmm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, c_values, a, b, alpha, beta, stream)

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

__INFINI_C infiniStatus_t infiniopDestroySDDMMDescriptor(infiniopSDDMMDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        delete reinterpret_cast<const op::sddmm::NAMESPACE::Descriptor *>(desc); \
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
