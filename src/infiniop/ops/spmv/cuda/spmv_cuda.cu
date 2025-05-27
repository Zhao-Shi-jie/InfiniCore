#include "spmv_cuda.cuh"
#include "spmv_cuda_internal.cuh"
#include "../../../devices/cuda/cuda_handle.cuh"

namespace op::spmv::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;

    Opaque(const std::shared_ptr<device::cuda::Handle::Internal> &internal)
        : internal(internal) {}
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    SparseFormat format) {

    auto info_result = SpMVInfo::create(y_desc, x_desc, values_desc, row_indices_desc, col_indices_desc, format);
    CHECK_RESULT(info_result);

    auto cuda_handle = reinterpret_cast<device::cuda::Handle *>(handle);
    auto opaque = new Opaque(cuda_handle->internal());

    *desc_ptr = new Descriptor(info_result.take(), opaque, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template<typename T>
infiniStatus_t launchSpMVKernel(
    const SpMVInfo &info,
    void *y,
    const void *x,
    const void *values,
    const void *row_indices,
    const void *col_indices,
    cudaStream_t stream) {

    const int BLOCK_SIZE = 256;
    dim3 blocks, threads(BLOCK_SIZE);

    // Initialize the output vector to zeros
    CHECK_CUDA(cudaMemsetAsync(y, 0, info.num_rows * sizeof(T), stream));

    if (info.format == SparseFormat::CSR) {
        blocks.x = (info.num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        spmv_csr_kernel<T><<<blocks, threads, 0, stream>>>(
            info.num_rows,
            static_cast<const int*>(row_indices),
            static_cast<const int*>(col_indices),
            static_cast<const T*>(values),
            static_cast<const T*>(x),
            static_cast<T*>(y)
        );
    } else if (info.format == SparseFormat::COO) {
        blocks.x = (info.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
        spmv_coo_kernel<T><<<blocks, threads, 0, stream>>>(
            info.nnz,
            static_cast<const int*>(row_indices),
            static_cast<const int*>(col_indices),
            static_cast<const T*>(values),
            static_cast<const T*>(x),
            static_cast<T*>(y)
        );
    }
    
    CHECK_LAST_CUDA_ERROR();
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *values,
    const void *row_indices,
    const void *col_indices,
    void *stream) const {

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launchSpMVKernel<half>(_info, y, x, values, row_indices, col_indices, cuda_stream);
    case INFINI_DTYPE_F32:
        return launchSpMVKernel<float>(_info, y, x, values, row_indices, col_indices, cuda_stream);
    case INFINI_DTYPE_F64:
        return launchSpMVKernel<double>(_info, y, x, values, row_indices, col_indices, cuda_stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::spmv::cuda
