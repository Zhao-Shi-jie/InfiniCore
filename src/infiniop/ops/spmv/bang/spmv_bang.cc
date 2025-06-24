#include "spmv_bang.h"

#include <iostream>
#include <vector>

#include "../../../devices/bang/bang_handle.h"
#include "../../../devices/bang/common_bang.h"
#include "../info.h"

__C void spmv_csr(int num_rows, int num_cols, int nnz, int *row_ptr,
                  int *col_indices, float *values, float *x, float *y);

namespace op::spmv::bang {
struct Descriptor::Opaque {
  std::shared_ptr<device::bang::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr, size_t num_cols,
                                  size_t num_rows, size_t nnz,
                                  infiniDtype_t dtype) {
  auto handle = reinterpret_cast<device::bang::cambricon::Handle *>(handle_);

  // currently only float32 supported
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
  // do basic validation
  auto validation_result =
      validateSpMVCSR(y, x, values, row_ptr, col_indices, _dtype);
  CHECK_OR_RETURN(validation_result == INFINI_STATUS_SUCCESS,
                  validation_result);
  std::cout << "SpMV validation passed" << std::endl;

  CNRT_CHECK(cnrtSetDevice(3));
  cnrtQueue_t queue;
  if (stream == nullptr || stream == NULL) {
    CNRT_CHECK(cnrtQueueCreate(&queue));
  } else {
    queue = (cnrtQueue_t)stream;
  }
  cnrtDim3_t dim = {128, 1, 1};
  cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;

  int num_rows = static_cast<int>(_info.num_rows);
  int num_cols = static_cast<int>(_info.num_cols);
  int nnz = static_cast<int>(_info.nnz);

  int *d_row_ptr = const_cast<int *>(static_cast<const int *>(row_ptr));
  int *d_col_indices = const_cast<int *>(static_cast<const int *>(col_indices));
  float *d_values = const_cast<float *>(static_cast<const float *>(values));
  float *d_x = const_cast<float *>(static_cast<const float *>(x));
  float *d_y = static_cast<float *>(y);
  // 打印 x 数组内容
  std::cout << "Printing x array (first 10 elements):" << std::endl;
  std::vector<float> host_x(std::min(num_cols, 10));  // 只打印前10个元素
  CNRT_CHECK(cnrtMemcpy(host_x.data(), d_x, host_x.size() * sizeof(float),
                        CNRT_MEM_TRANS_DIR_DEV2HOST));
  for (size_t i = 0; i < host_x.size(); ++i) {
    std::cout << "x[" << i << "] = " << host_x[i] << std::endl;
  }
  void *args[8];
  args[0] = &num_rows;
  args[1] = &num_cols;
  args[2] = &nnz;
  args[3] = &d_row_ptr;      // d_Ap
  args[4] = &d_col_indices;  // d_Aj
  args[5] = &d_values;       // d_Ax
  args[6] = &d_x;            // d_x
  args[7] = &d_y;            // d_y

  std::cout << "SpMV calculation started" << std::endl;
  // 启动 kernel
  CNRT_CHECK(cnrtInvokeKernel((void *)&spmv_csr, dim, ktype, args, 0, queue));

  CNRT_CHECK(cnrtQueueSync(queue));

  return INFINI_STATUS_SUCCESS;
}

}  // namespace op::spmv::bang