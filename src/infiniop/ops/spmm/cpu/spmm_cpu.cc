#include "spmm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::spmm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SpMMInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        a_desc->indexDtype(),
        result.take(),
        a_desc,
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
void calculateCsr(
    const SpMMInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *c,
    const void *b,
    float alpha,
    float beta) {
    auto values = reinterpret_cast<const Tdata *>(a_desc->values());
    auto crow_indices = reinterpret_cast<const Tindex *>(a_desc->crowIndices());
    auto col_indices = reinterpret_cast<const Tindex *>(a_desc->colIndices());
    auto b_data = reinterpret_cast<const Tdata *>(b);
    auto c_data = reinterpret_cast<Tdata *>(c);

#pragma omp parallel for
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(info.m); ++row) {
        for (size_t col = 0; col < info.n; ++col) {
            auto c_offset = row * info.c_matrix.row_stride + col * info.c_matrix.col_stride;
            float acc = 0;
            for (Tindex ptr = crow_indices[row]; ptr < crow_indices[row + 1]; ++ptr) {
                auto k = static_cast<size_t>(col_indices[ptr]);
                auto b_offset = k * info.b_matrix.row_stride + col * info.b_matrix.col_stride;
                acc += utils::cast<float>(values[ptr]) * utils::cast<float>(b_data[b_offset]);
            }
            if (beta == 0) {
                c_data[c_offset] = utils::cast<Tdata>(alpha * acc);
            } else {
                c_data[c_offset] = utils::cast<Tdata>(alpha * acc + beta * utils::cast<float>(c_data[c_offset]));
            }
        }
    }
}

template <typename Tdata, typename Tindex>
void calculateEll(
    const SpMMInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *c,
    const void *b,
    float alpha,
    float beta) {
    auto values = reinterpret_cast<const Tdata *>(a_desc->values());
    auto col_indices = reinterpret_cast<const Tindex *>(a_desc->colIndices());
    auto b_data = reinterpret_cast<const Tdata *>(b);
    auto c_data = reinterpret_cast<Tdata *>(c);

#pragma omp parallel for
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(info.m); ++row) {
        for (size_t col = 0; col < info.n; ++col) {
            auto c_offset = row * info.c_matrix.row_stride + col * info.c_matrix.col_stride;
            float acc = 0;
            for (size_t slot = 0; slot < info.ell_width; ++slot) {
                size_t ptr = static_cast<size_t>(row) * info.ell_width + slot;
                auto value = utils::cast<float>(values[ptr]);
                if (value == 0.0f) {
                    continue;
                }
                auto k = static_cast<size_t>(col_indices[ptr]);
                auto b_offset = k * info.b_matrix.row_stride + col * info.b_matrix.col_stride;
                acc += value * utils::cast<float>(b_data[b_offset]);
            }
            if (beta == 0) {
                c_data[c_offset] = utils::cast<Tdata>(alpha * acc);
            } else {
                c_data[c_offset] = utils::cast<Tdata>(alpha * acc + beta * utils::cast<float>(c_data[c_offset]));
            }
        }
    }
}

template <typename Tdata, typename Tindex>
void calculateSell(
    const SpMMInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *c,
    const void *b,
    float alpha,
    float beta) {
    auto values = reinterpret_cast<const Tdata *>(a_desc->values());
    auto slice_offsets = reinterpret_cast<const Tindex *>(a_desc->sliceOffsets());
    auto col_indices = reinterpret_cast<const Tindex *>(a_desc->colIndices());
    auto row_indices = reinterpret_cast<const Tindex *>(a_desc->rowIndices());
    auto b_data = reinterpret_cast<const Tdata *>(b);
    auto c_data = reinterpret_cast<Tdata *>(c);
    bool has_row_permutation = info.format == INFINIOP_SPMAT_FORMAT_SELL_SIGMA_C;

#pragma omp parallel for
    for (ptrdiff_t storage_row = 0; storage_row < static_cast<ptrdiff_t>(info.m); ++storage_row) {
        size_t slice = static_cast<size_t>(storage_row) / info.slice_height;
        size_t row_in_slice = static_cast<size_t>(storage_row) - slice * info.slice_height;
        size_t row = has_row_permutation ? static_cast<size_t>(row_indices[storage_row]) : static_cast<size_t>(storage_row);
        size_t slice_begin = static_cast<size_t>(slice_offsets[slice]);
        size_t slice_end = static_cast<size_t>(slice_offsets[slice + 1]);
        size_t slice_width = (slice_end - slice_begin) / info.slice_height;

        for (size_t col = 0; col < info.n; ++col) {
            auto c_offset = row * info.c_matrix.row_stride + col * info.c_matrix.col_stride;
            float acc = 0;
            for (size_t slot = 0; slot < slice_width; ++slot) {
                size_t ptr = slice_begin + slot * info.slice_height + row_in_slice;
                auto value = utils::cast<float>(values[ptr]);
                if (value == 0.0f) {
                    continue;
                }
                auto k = static_cast<size_t>(col_indices[ptr]);
                auto b_offset = k * info.b_matrix.row_stride + col * info.b_matrix.col_stride;
                acc += value * utils::cast<float>(b_data[b_offset]);
            }
            if (beta == 0) {
                c_data[c_offset] = utils::cast<Tdata>(alpha * acc);
            } else {
                c_data[c_offset] = utils::cast<Tdata>(alpha * acc + beta * utils::cast<float>(c_data[c_offset]));
            }
        }
    }
}

template <typename Tdata>
infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const SpMMInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *c,
    const void *b,
    float alpha,
    float beta) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        switch (info.format) {
        case INFINIOP_SPMAT_FORMAT_CSR:
            calculateCsr<Tdata, int32_t>(info, a_desc, c, b, alpha, beta);
            return INFINI_STATUS_SUCCESS;
        case INFINIOP_SPMAT_FORMAT_ELL:
            calculateEll<Tdata, int32_t>(info, a_desc, c, b, alpha, beta);
            return INFINI_STATUS_SUCCESS;
        case INFINIOP_SPMAT_FORMAT_SELL:
        case INFINIOP_SPMAT_FORMAT_SELL_SIGMA_C:
            calculateSell<Tdata, int32_t>(info, a_desc, c, b, alpha, beta);
            return INFINI_STATUS_SUCCESS;
        default:
            return INFINI_STATUS_BAD_PARAM;
        }
    case INFINI_DTYPE_I64:
        switch (info.format) {
        case INFINIOP_SPMAT_FORMAT_CSR:
            calculateCsr<Tdata, int64_t>(info, a_desc, c, b, alpha, beta);
            return INFINI_STATUS_SUCCESS;
        case INFINIOP_SPMAT_FORMAT_ELL:
            calculateEll<Tdata, int64_t>(info, a_desc, c, b, alpha, beta);
            return INFINI_STATUS_SUCCESS;
        case INFINIOP_SPMAT_FORMAT_SELL:
        case INFINIOP_SPMAT_FORMAT_SELL_SIGMA_C:
            calculateSell<Tdata, int64_t>(info, a_desc, c, b, alpha, beta);
            return INFINI_STATUS_SUCCESS;
        default:
            return INFINI_STATUS_BAD_PARAM;
        }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<fp16_t>(_index_dtype, _info, _a_desc, c, b, alpha, beta);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<bf16_t>(_index_dtype, _info, _a_desc, c, b, alpha, beta);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_index_dtype, _info, _a_desc, c, b, alpha, beta);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::spmm::cpu
