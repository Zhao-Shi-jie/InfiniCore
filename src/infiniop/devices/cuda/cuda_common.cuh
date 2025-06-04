#ifndef __INFINIOP_CUDA_COMMON_CUH__
#define __INFINIOP_CUDA_COMMON_CUH__

#include "cuda_handle.cuh"
#include "infinicore.h"
#include <cuda_runtime.h>
#include <cusparse.h>

namespace device::cuda {

cudnnDataType_t getCudnnDtype(infiniDtype_t dt);

} // namespace device::cuda


// cuSPARSE错误检查宏
#define CHECK_CUSPARSE(call)                                            \
    do {                                                                \
        cusparseStatus_t err = call;                                    \
        if (err != CUSPARSE_STATUS_SUCCESS) {                          \
            return INFINI_STATUS_INTERNAL_ERROR;                       \
        }                                                               \
    } while (0)

#endif // __INFINIOP_CUDA_COMMON_CUH__
