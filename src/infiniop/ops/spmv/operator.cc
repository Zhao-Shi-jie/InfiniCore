// #include "../../../infiniop.h"
// #include "info.h"
// #include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/spmv.h"
// 包含各设备的实现头文件
#ifdef ENABLE_CPU_API
#include "cpu/spmv_cpu.h"
#endif

#ifdef ENABLE_CUDA_API
#include "cuda/spmv_cuda.h"
#endif

__C infiniStatus_t infiniopSpMV_csr(
    infiniopHandle_t handle,
    void *y,
    const void *x,
    const void *values,
    const void *row_indices,
    const void *col_indices,
    size_t num_rows,
    size_t num_cols,
    size_t nnz,
    infiniDtype_t dtype,
    void *stream) {

#define SPMV_CSR(CASE, NAMESPACE)                           \
    case CASE:                                              \
        return op::spmv::NAMESPACE::spmv_csr(               \
            handle, y, x, values, row_indices, col_indices, \
            num_rows, num_cols, nnz, dtype, stream)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        SPMV_CSR(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        SPMV_CSR(INFINI_DEVICE_NVIDIA, cuda);
        // SPMV_CSR(INFINI_DEVICE_MOORE, cuda);
        // SPMV_CSR(INFINI_DEVICE_ILUVATAR, cuda);
        // SPMV_CSR(INFINI_DEVICE_METAX, cuda);
        // SPMV_CSR(INFINI_DEVICE_SUGON, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef SPMV_CSR
}
