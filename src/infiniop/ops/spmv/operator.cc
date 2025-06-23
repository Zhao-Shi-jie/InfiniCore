#include "../../handle.h"
#include "infiniop/ops/spmv.h"
#include "spmv.h"

#ifdef ENABLE_CPU_API
#include "cpu/spmv_cpu.h"
#endif

#ifdef ENABLE_CUDA_API
#include "cuda/spmv_cuda.cuh"
#endif

__C infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    size_t num_cols,
    size_t num_rows,
    size_t nnz,
    infiniDtype_t dtype) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::spmv::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::spmv::NAMESPACE::Descriptor **>(desc_ptr), \
            num_cols, num_rows, nnz, dtype)

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

__C infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *y,
    const void *x,
    const void *values,
    const void *row_ptr,
    const void *col_indices,
    void *stream) {

    printf("=== C Interface Debug ===\n");
    printf("y ptr: %p\n", y);
    printf("x ptr: %p\n", x);
    printf("values ptr: %p\n", values);
    printf("row_ptr ptr: %p\n", row_ptr);
    printf("col_indices ptr: %p\n", col_indices);
    printf("stream ptr: %p\n", stream);

    // // 尝试读取row_ptr的前几个值
    // if (row_ptr) {
    //     int32_t *row_ptr_int = (int32_t *)row_ptr;
    //     printCudaArray<int>(row_ptr, 100 + 1);
    //     printCudaArray<int>(row_ptr_int, 100 + 1);
    // }

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::spmv::NAMESPACE::Descriptor *>(desc) \
            ->calculate(y, x, values, row_ptr, col_indices, stream)

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

__C infiniStatus_t infiniopDestroySpMVDescriptor(infiniopSpMVDescriptor_t desc) {

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

// 简化的直接API实现
// __C infiniStatus_t infiniopSpMV_csr(
//     infiniopHandle_t handle,
//     void *y,
//     const void *x,
//     const void *values,
//     const void *row_indices,
//     const void *col_indices,
//     size_t num_cols,
//     infiniDtype_t dtype,
//     void *stream) {

//     // 使用描述符API实现
//     infiniopSpMVDescriptor_t desc;
//     auto status = infiniopCreateSpMVDescriptor(handle, &desc, num_cols, row_indices, dtype);
//     if (status != INFINI_STATUS_SUCCESS) {
//         return status;
//     }

//     status = infiniopSpMV(desc, y, x, values, row_indices, col_indices, stream);
//     infiniopDestroySpMVDescriptor(desc);

//     return status;
// }
