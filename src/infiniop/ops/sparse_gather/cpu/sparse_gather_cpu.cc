#include "sparse_gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::sparse_gather::cpu {

Descriptor::~Descriptor() = default;

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
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    auto result = SparseGatherInfo::create(output_desc, pattern_desc, input_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        pattern_desc->indicesDesc()->dtype(),
        result.take(),
        pattern_desc,
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
static void calculateSparseGather(
    const SparseGatherInfo &info,
    infiniopSpVecDescriptor_t pattern_desc,
    void *output,
    const void *input) {
    auto indices = reinterpret_cast<const Tindex *>(pattern_desc->indices());
    auto out = reinterpret_cast<Tdata *>(output);
    auto in = reinterpret_cast<const Tdata *>(input);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(info.nnz); ++i) {
        auto index = indices[i];
        auto out_offset = i * info.output_stride;
        if (index >= 0 && static_cast<size_t>(index) < info.input_vector.size) {
            out[out_offset] = in[static_cast<size_t>(index) * info.input_vector.stride];
        } else {
            out[out_offset] = utils::cast<Tdata>(0.0f);
        }
    }
}

template <typename Tdata>
static infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const SparseGatherInfo &info,
    infiniopSpVecDescriptor_t pattern_desc,
    void *output,
    const void *input) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        calculateSparseGather<Tdata, int32_t>(info, pattern_desc, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I64:
        calculateSparseGather<Tdata, int64_t>(info, pattern_desc, output, input);
        return INFINI_STATUS_SUCCESS;
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
    (void)stream;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<fp16_t>(_index_dtype, _info, _pattern_desc, output, input);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<bf16_t>(_index_dtype, _info, _pattern_desc, output, input);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_index_dtype, _info, _pattern_desc, output, input);
    case INFINI_DTYPE_F64:
        return calculateByIndex<double>(_index_dtype, _info, _pattern_desc, output, input);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::sparse_gather::cpu
