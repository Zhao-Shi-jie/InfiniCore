#ifndef __SPMV_CUDA_H__
#define __SPMV_CUDA_H__

#include "../../../devices/cuda/cuda_kernel_common.cuh"
#include <cuda_fp16.h>

namespace op::spmv::cuda {

// CSR format kernel
template<typename T>
__global__ void spmv_csr_kernel(
    size_t num_rows,
    const int *row_ptrs,
    const int *col_indices,
    const T *values,
    const T *x,
    T *y) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        const int row_start = row_ptrs[row];
        const int row_end = row_ptrs[row + 1];
        
        T sum = T(0);
        for (int i = row_start; i < row_end; i++) {
            const int col = col_indices[i];
            if constexpr (std::is_same_v<T, half>) {
                sum = __hadd(sum, __hmul(values[i], x[col]));
            } else {
                sum += values[i] * x[col];
            }
        }
        
        y[row] = sum;
    }
}

// COO format kernel
template<typename T>
__global__ void spmv_coo_kernel(
    size_t nnz,
    const int *row_indices,
    const int *col_indices,
    const T *values,
    const T *x,
    T *y) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        const int row = row_indices[idx];
        const int col = col_indices[idx];
        
        if constexpr (std::is_same_v<T, half>) {
            atomicAdd(&y[row], __hmul(values[idx], x[col]));
        } else {
            atomicAdd(&y[row], values[idx] * x[col]);
        }
    }
}

// Specialized half version for atomicAdd
template<>
__global__ void spmv_coo_kernel<half>(
    size_t nnz,
    const int *row_indices,
    const int *col_indices,
    const half *values,
    const half *x,
    half *y) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        const int row = row_indices[idx];
        const int col = col_indices[idx];
        
        // Convert to float for atomicAdd since half doesn't have direct atomic operations
        float val = __half2float(__hmul(values[idx], x[col]));
        atomicAdd(reinterpret_cast<float*>(&y[row]), val);
    }
}

} // namespace op::spmv::cuda

#endif // __SPMV_CUDA_H__
