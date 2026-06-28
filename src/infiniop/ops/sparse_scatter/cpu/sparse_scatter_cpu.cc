#include "sparse_scatter_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::sparse_scatter::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopSpVecDescriptor_t input_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    auto result = SparseScatterInfo::create(output_desc, input_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        input_desc->indicesDesc()->dtype(),
        result.take(),
        input_desc,
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
static void calculateSparseScatter(
    const SparseScatterInfo &info,
    infiniopSpVecDescriptor_t input_desc,
    void *output) {
    auto indices = reinterpret_cast<const Tindex *>(input_desc->indices());
    auto values = reinterpret_cast<const Tdata *>(input_desc->values());
    auto out = reinterpret_cast<Tdata *>(output);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(info.nnz); ++i) {
        auto index = indices[i];
        if (index >= 0 && static_cast<size_t>(index) < info.output_vector.size) {
            out[static_cast<size_t>(index) * info.output_vector.stride] = values[i];
        }
    }
}

template <typename Tdata>
static infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const SparseScatterInfo &info,
    infiniopSpVecDescriptor_t input_desc,
    void *output) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        calculateSparseScatter<Tdata, int32_t>(info, input_desc, output);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I64:
        calculateSparseScatter<Tdata, int64_t>(info, input_desc, output);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<fp16_t>(_index_dtype, _info, _input_desc, output);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<bf16_t>(_index_dtype, _info, _input_desc, output);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_index_dtype, _info, _input_desc, output);
    case INFINI_DTYPE_F64:
        return calculateByIndex<double>(_index_dtype, _info, _input_desc, output);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::sparse_scatter::cpu
