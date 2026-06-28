#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "sparse_gather_nvidia.cuh"

#include <cuda_fp16.h>

namespace op::sparse_gather::nvidia {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopSpVecDescriptor_t pattern_desc,
    infiniopTensorDescriptor_t input_desc,
    void *output,
    const void *input) {
    (void)output;
    (void)input;
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    auto result = SparseGatherInfo::create(output_desc, pattern_desc, input_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        pattern_desc->indicesDesc()->dtype(),
        result.take(),
        pattern_desc,
        0,
        new Opaque(),
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
INFINIOP_CUDA_KERNEL sparseGatherKernel(
    size_t nnz,
    size_t input_size,
    ptrdiff_t input_stride,
    ptrdiff_t output_stride,
    const Tindex *indices,
    Tdata *output,
    const Tdata *input) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }

    auto index = indices[i];
    auto out_offset = i * output_stride;
    if (index >= 0 && static_cast<size_t>(index) < input_size) {
        output[out_offset] = input[static_cast<size_t>(index) * input_stride];
    } else {
        output[out_offset] = static_cast<Tdata>(0);
    }
}

template <>
INFINIOP_CUDA_KERNEL sparseGatherKernel<__half, int32_t>(
    size_t nnz,
    size_t input_size,
    ptrdiff_t input_stride,
    ptrdiff_t output_stride,
    const int32_t *indices,
    __half *output,
    const __half *input) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    auto index = indices[i];
    auto out_offset = i * output_stride;
    output[out_offset] = (index >= 0 && static_cast<size_t>(index) < input_size)
                           ? input[static_cast<size_t>(index) * input_stride]
                           : __float2half(0.0f);
}

template <>
INFINIOP_CUDA_KERNEL sparseGatherKernel<__half, int64_t>(
    size_t nnz,
    size_t input_size,
    ptrdiff_t input_stride,
    ptrdiff_t output_stride,
    const int64_t *indices,
    __half *output,
    const __half *input) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    auto index = indices[i];
    auto out_offset = i * output_stride;
    output[out_offset] = (index >= 0 && static_cast<size_t>(index) < input_size)
                           ? input[static_cast<size_t>(index) * input_stride]
                           : __float2half(0.0f);
}

template <typename Tdata, typename Tindex>
static infiniStatus_t launch(
    const SparseGatherInfo &info,
    infiniopSpVecDescriptor_t pattern_desc,
    void *output,
    const void *input,
    void *stream) {
    constexpr size_t block = 256;
    auto grid = (info.nnz + block - 1) / block;
    sparseGatherKernel<Tdata, Tindex><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        info.nnz,
        info.input_vector.size,
        info.input_vector.stride,
        info.output_stride,
        reinterpret_cast<const Tindex *>(pattern_desc->indices()),
        reinterpret_cast<Tdata *>(output),
        reinterpret_cast<const Tdata *>(input));
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
static infiniStatus_t launchByIndex(
    infiniDtype_t index_dtype,
    const SparseGatherInfo &info,
    infiniopSpVecDescriptor_t pattern_desc,
    void *output,
    const void *input,
    void *stream) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        return launch<Tdata, int32_t>(info, pattern_desc, output, input, stream);
    case INFINI_DTYPE_I64:
        return launch<Tdata, int64_t>(info, pattern_desc, output, input, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return launchByIndex<__half>(_index_dtype, _info, _pattern_desc, output, input, stream);
    case INFINI_DTYPE_F32:
        return launchByIndex<float>(_index_dtype, _info, _pattern_desc, output, input, stream);
    case INFINI_DTYPE_F64:
        return launchByIndex<double>(_index_dtype, _info, _pattern_desc, output, input, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::sparse_gather::nvidia
