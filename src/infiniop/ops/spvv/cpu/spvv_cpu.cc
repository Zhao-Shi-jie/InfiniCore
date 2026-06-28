#include "spvv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::spvv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopSpVecDescriptor_t a_desc,
    infiniopTensorDescriptor_t x_desc,
    const void *x) {
    (void)x;
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SpVVInfo::create(y_desc, a_desc, x_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        a_desc->indicesDesc()->dtype(),
        result.take(),
        a_desc,
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
void calculate(
    const SpVVInfo &info,
    infiniopSpVecDescriptor_t a_desc,
    void *y,
    const void *x) {
    auto values = reinterpret_cast<const Tdata *>(a_desc->values());
    auto indices = reinterpret_cast<const Tindex *>(a_desc->indices());
    auto x_data = reinterpret_cast<const Tdata *>(x);
    auto y_data = reinterpret_cast<Tdata *>(y);

    float acc = 0;
    for (size_t i = 0; i < info.nnz; ++i) {
        auto index = static_cast<size_t>(indices[i]);
        acc += utils::cast<float>(values[i]) * utils::cast<float>(x_data[index * info.x_vector.stride]);
    }

    *y_data = utils::cast<Tdata>(acc);
}

template <typename Tdata>
infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const SpVVInfo &info,
    infiniopSpVecDescriptor_t a_desc,
    void *y,
    const void *x) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        calculate<Tdata, int32_t>(info, a_desc, y, x);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I64:
        calculate<Tdata, int64_t>(info, a_desc, y, x);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *,
    size_t workspace_size,
    void *y,
    const void *x,
    void *) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<fp16_t>(_index_dtype, _info, _a_desc, y, x);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<bf16_t>(_index_dtype, _info, _a_desc, y, x);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_index_dtype, _info, _a_desc, y, x);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::spvv::cpu
