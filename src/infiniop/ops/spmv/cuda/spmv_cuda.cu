#include "spmv_cuda.h"
#include "../../../devices/cuda/cuda_common.cuh"
#include "../../../devices/cuda/cuda_kernel_common.cuh"
#include "../info.h"
#include <cusparse.h>

namespace op::spmv::cuda {

infiniStatus_t spmv_csr(
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

    // 参数验证
    auto validation_result = validateSpMVCSR(
        y, x, values, row_indices, col_indices,
        num_rows, num_cols, nnz, dtype);
    CHECK_OR_RETURN(validation_result == INFINI_STATUS_SUCCESS, validation_result);

    // 获取CUDA流
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    // 创建cuSPARSE句柄
    cusparseHandle_t cusparse_handle;
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
    CHECK_CUSPARSE(cusparseSetStream(cusparse_handle, cuda_stream));

    // 确定CUDA数据类型
    cudaDataType cuda_dtype;
    switch (dtype) {
        case INFINI_DTYPE_F32:
            cuda_dtype = CUDA_R_32F;
            break;
        case INFINI_DTYPE_F64:
            cuda_dtype = CUDA_R_64F;
            break;
        case INFINI_DTYPE_F16:
            cuda_dtype = CUDA_R_16F;
            break;
        default:
            cusparseDestroy(cusparse_handle);
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // 创建稀疏矩阵描述符
    cusparseSpMatDescr_t mat_descr;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &mat_descr,
        num_rows, num_cols, nnz,
        const_cast<void *>(row_indices), 
        const_cast<void *>(col_indices), 
        const_cast<void *>(values),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
        CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));

    // 创建稠密向量描述符
    cusparseDnVecDescr_t vec_x, vec_y;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x, num_cols, const_cast<void *>(x), cuda_dtype));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y, num_rows, y, cuda_dtype));

    // 设置计算参数
    const float alpha = 1.0f, beta = 0.0f;

    // 计算所需的缓冲区大小
    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_descr, vec_x, &beta, vec_y, cuda_dtype,
        CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));

    // 分配缓冲区
    void *external_buffer = nullptr;
    if (buffer_size > 0) {
        CHECK_CUDA(cudaMalloc(&external_buffer, buffer_size));
    }

    // 执行SpMV计算
    auto result = cusparseSpMV(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_descr, vec_x, &beta, vec_y, cuda_dtype,
        CUSPARSE_SPMV_ALG_DEFAULT, external_buffer);

    // 清理资源
    if (external_buffer) {
        cudaFree(external_buffer);
    }
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
    cusparseDestroySpMat(mat_descr);
    cusparseDestroy(cusparse_handle);

    CHECK_CUSPARSE(result);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv::cuda
