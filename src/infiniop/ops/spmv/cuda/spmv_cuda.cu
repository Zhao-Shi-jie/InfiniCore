#include "../../../devices/cuda/cuda_common.cuh"
#include "../../../devices/cuda/cuda_handle.cuh"
#include "../../../devices/cuda/cuda_kernel_common.cuh"
#include "../info.h"
#include "spmv_cuda.cuh"
#include <cstdint>
#include <cusparse.h>

// Add this before the calculate function or in a header file

template <typename T>
void printCudaArray(const void *device_ptr, size_t count,
                    const char *name = "") {
  if (count == 0) {
    printf("%s: [empty]\n", name);
    return;
  }

  // Allocate host memory
  T *host_data = new T[count];

  // Copy from device to host
  cudaError_t err = cudaMemcpy(host_data, device_ptr, count * sizeof(T),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("%s: Error copying from device: %s\n", name,
           cudaGetErrorString(err));
    delete[] host_data;
    return;
  }

  // Print first few elements (limit output for readability)
  size_t print_count = std::min(count, size_t(10));
  printf("%s: [", name);
  for (size_t i = 0; i < print_count; ++i) {
    if (i > 0)
      printf(", ");
    if constexpr (std::is_integral<T>::value) {
      printf("%d", static_cast<int>(host_data[i]));
    } else {
      printf("%.6f", static_cast<double>(host_data[i]));
    }
  }
  if (count > print_count) {
    printf(", ... (%zu more)", count - print_count);
  }
  printf("]\n");

  delete[] host_data;
}

namespace op::spmv::cuda {

struct Descriptor::Opaque {
  std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr, size_t num_cols,
                                  size_t num_rows, size_t nnz,
                                  infiniDtype_t dtype) {

  auto handle = reinterpret_cast<device::cuda::nvidia::Handle *>(handle_);

  // 当前仅支持单精度
  if (dtype != INFINI_DTYPE_F32) {
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
  }

  auto result = SpMVInfo::create(num_cols, num_rows, nnz);
  CHECK_RESULT(result);

  *desc_ptr =
      new Descriptor(dtype, result.take(), new Opaque{handle->internal()},
                     handle->device, handle->device_id);
  return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *y, const void *x, const void *values,
                                     const void *row_ptr,
                                     const void *col_indices,
                                     void *stream) const {

  // 计算之前，打印参数的真实数据

  printf("--------------kernel 内：SpMV CSR Parameters: -------------\n");
  printf("num_rows: %zu, num_cols: %zu, nnz: %zu\n", _info.num_rows,
         _info.num_cols, _info.nnz);

  printf("  sizeof(int): %zu\n", sizeof(int));
  printf("  sizeof(int32_t): %zu\n", sizeof(int32_t));
  printf("  sizeof(size_t): %zu\n", sizeof(size_t));

  // 打印指针下的实际数据
  printf("x: ");
  printCudaArray<float>(x, _info.num_cols); // 打印x向量
  printf("values: ");
  printCudaArray<float>(values, _info.nnz); // 打印非零元素
  printf("row_ptr: ");
  printCudaArray<int32_t>(row_ptr, _info.num_rows + 1); // 打印行指针
  printf("col_indices: ");
  printCudaArray<int32_t>(col_indices, _info.nnz); // 打印列索引

  // 参数验证
  auto validation_result =
      validateSpMVCSR(y, x, values, row_ptr, col_indices, _dtype);
  CHECK_OR_RETURN(validation_result == INFINI_STATUS_SUCCESS,
                  validation_result);

  // 仅支持单精度
  cudaDataType cuda_dtype = CUDA_R_32F;
  const float alpha = 1.0f, beta = 0.0f;

  CHECK_STATUS(_opaque->internal->useCusparse(
      (cudaStream_t)stream, [&](cusparseHandle_t cusparse_handle) {
        // 创建稀疏矩阵描述符
        cusparseSpMatDescr_t mat_descr;
        CHECK_CUSPARSE(
            cusparseCreateCsr(&mat_descr,
                              _info.num_rows,              // 行数
                              _info.num_cols,              // 列数
                              _info.nnz,                   // 非零元素数量
                              const_cast<void *>(row_ptr), // 行指针
                              const_cast<void *>(col_indices), // 列索引
                              const_cast<void *>(values), // 非零元素值
                              CUSPARSE_INDEX_32I,         // 索引类型
                              CUSPARSE_INDEX_32I,         // 列索引类型
                              CUSPARSE_INDEX_BASE_ZERO,   // 索引基准
                              cuda_dtype));

        // 创建稠密向量描述符
        cusparseDnVecDescr_t vec_x, vec_y;
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x,
                                           _info.num_cols,        // 列数
                                           const_cast<void *>(x), // 数据指针
                                           cuda_dtype));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y,
                                           _info.num_rows, // 行数
                                           y,              // 数据指针
                                           cuda_dtype));

        // 计算所需的缓冲区大小
        size_t buffer_size = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            cusparse_handle,                  // 句柄
            CUSPARSE_OPERATION_NON_TRANSPOSE, // 操作类型
            &alpha,                           // 标量alpha
            mat_descr,                        // 稀疏矩阵描述符
            vec_x,                            // 稠密向量描述符
            &beta,                            // 标量beta
            vec_y,                            // 稠密向量描述符
            cuda_dtype,                       // 数据类型
            CUSPARSE_SPMV_ALG_DEFAULT,        // 算法选择
            &buffer_size));                   // 缓冲区大小指针

        // 分配缓冲区
        void *external_buffer = nullptr;
        if (buffer_size > 0) {
          CHECK_CUDA(cudaMalloc(&external_buffer, buffer_size));
        }

        // 执行SpMV计算
        auto result =
            cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, mat_descr, vec_x, &beta, vec_y, cuda_dtype,
                         CUSPARSE_SPMV_ALG_DEFAULT, external_buffer);

        // 清理资源
        if (external_buffer) {
          cudaFree(external_buffer);
        }
        cusparseDestroyDnVec(vec_x);
        cusparseDestroyDnVec(vec_y);
        cusparseDestroySpMat(mat_descr);

        CHECK_CUSPARSE(result);
        return INFINI_STATUS_SUCCESS;
      }));

  return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv::cuda
