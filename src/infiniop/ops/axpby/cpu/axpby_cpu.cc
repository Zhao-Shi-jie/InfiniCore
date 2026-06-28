#include "axpby_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::axpby::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopSpVecDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = AxpbyInfo::create(x_desc, y_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        x_desc,
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
static void calculateAxpby(
    const AxpbyInfo &info,
    infiniopSpVecDescriptor_t x_desc,
    void *y,
    float alpha,
    float beta) {
    auto x_values = reinterpret_cast<const Tdata *>(x_desc->values());
    auto x_indices = reinterpret_cast<const Tindex *>(x_desc->indices());
    auto y_data = reinterpret_cast<Tdata *>(y);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(info.n); ++i) {
        auto y_offset = i * info.incy;
        y_data[y_offset] = utils::cast<Tdata>(beta * utils::cast<float>(y_data[y_offset]));
    }

    for (size_t i = 0; i < info.nnz; ++i) {
        auto index = static_cast<size_t>(x_indices[i]);
        auto y_offset = static_cast<ptrdiff_t>(index) * info.incy;
        auto result = utils::cast<float>(y_data[y_offset]) + alpha * utils::cast<float>(x_values[i]);
        y_data[y_offset] = utils::cast<Tdata>(result);
    }
}

template <typename Tdata>
static infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const AxpbyInfo &info,
    infiniopSpVecDescriptor_t x_desc,
    void *y,
    float alpha,
    float beta) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        calculateAxpby<Tdata, int32_t>(info, x_desc, y, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I64:
        calculateAxpby<Tdata, int64_t>(info, x_desc, y, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    float alpha,
    float beta,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<fp16_t>(_x_desc->indicesDesc()->dtype(), _info, _x_desc, y, alpha, beta);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<bf16_t>(_x_desc->indicesDesc()->dtype(), _info, _x_desc, y, alpha, beta);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_x_desc->indicesDesc()->dtype(), _info, _x_desc, y, alpha, beta);
    case INFINI_DTYPE_F64:
        return calculateByIndex<double>(_x_desc->indicesDesc()->dtype(), _info, _x_desc, y, alpha, beta);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::axpby::cpu
