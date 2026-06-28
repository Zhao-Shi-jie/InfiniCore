#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/sparse_gather.h"

#ifdef ENABLE_CPU_API
#include "cpu/sparse_gather_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/sparse_gather_nvidia.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/sparse_gather_bang.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/sparse_gather_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateSparseGatherDescriptor(
    infiniopHandle_t handle,
    infiniopSparseGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopSpVecDescriptor_t pattern_desc,
    infiniopTensorDescriptor_t input_desc,
    void *output,
    const void *input) {
#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::sparse_gather::NAMESPACE::Descriptor::create(                     \
            handle,                                                                  \
            reinterpret_cast<op::sparse_gather::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                             \
            pattern_desc,                                                            \
            input_desc,                                                              \
            output,                                                                  \
            input)

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

__INFINI_C infiniStatus_t infiniopGetSparseGatherWorkspaceSize(infiniopSparseGatherDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                               \
    case CASE:                                                                                             \
        *size = reinterpret_cast<const op::sparse_gather::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopSparseGather(
    infiniopSparseGatherDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                          \
        return reinterpret_cast<const op::sparse_gather::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, stream)

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

__INFINI_C infiniStatus_t infiniopDestroySparseGatherDescriptor(infiniopSparseGatherDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        delete reinterpret_cast<const op::sparse_gather::NAMESPACE::Descriptor *>(desc); \
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
