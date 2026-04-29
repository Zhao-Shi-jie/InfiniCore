#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/spmm.h"

#ifdef ENABLE_CPU_API
#include "cpu/spmm_cpu.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateSpmmDescriptor(
    infiniopHandle_t handle,
    infiniopSpmmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t values_desc,
    size_t rows,
    size_t cols) {

#define CREATE(CASE, NAMESPACE)                                              \
    case CASE:                                                               \
        return op::spmm::NAMESPACE::Descriptor::create(                      \
            handle,                                                          \
            reinterpret_cast<op::spmm::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc, b_desc, values_desc, rows, cols)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t
infiniopGetSpmmWorkspaceSize(
    infiniopSpmmDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<const op::spmm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopSpmm(
    infiniopSpmmDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *c,
    const void *row_offsets,
    const void *col_indices,
    const void *values,
    const void *b,
    float alpha,
    float beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<const op::spmm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                              \
                        c, row_offsets, col_indices, values, b,                 \
                        alpha, beta, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t
infiniopDestroySpmmDescriptor(infiniopSpmmDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        delete reinterpret_cast<const op::spmm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
