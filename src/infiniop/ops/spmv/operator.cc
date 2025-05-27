#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/spmv.h"
#include "info.h"

#ifdef ENABLE_CPU_API
#include "cpu/spmv_cpu.h"
#endif

#ifdef ENABLE_CUDA_API
#include "cuda/spmv_cuda.cuh"
#endif

__C infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    int sparse_format) {

    op::spmv::SparseFormat format = static_cast<op::spmv::SparseFormat>(sparse_format);

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::spmv::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::spmv::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                         \
            x_desc,                                                         \
            values_desc,                                                    \
            row_indices_desc,                                               \
            col_indices_desc,                                               \
            format);

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetSpMVWorkspaceSize(
    infiniopSpMVDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                \
    case CASE:                                                                              \
        *size = reinterpret_cast<op::spmv::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        GET(INFINI_DEVICE_NVIDIA, cuda)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *values,
    const void *row_indices,
    const void *col_indices,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::spmv::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, x, values,               \
                        row_indices, col_indices, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroySpMVDescriptor(infiniopSpMVDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::spmv::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        DELETE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
