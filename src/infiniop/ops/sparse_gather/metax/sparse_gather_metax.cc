#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"
#include "../../../devices/metax/metax_ht2mc.h"
#include "sparse_gather_metax.h"

#include <cstdint>

namespace op::sparse_gather::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    hcsparseSpVecDescr_t vec_x = nullptr;
    hcsparseDnVecDescr_t vec_y = nullptr;

    explicit Opaque(std::shared_ptr<device::metax::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (vec_y != nullptr) {
            hcsparseDestroyDnVec(vec_y);
        }
        if (vec_x != nullptr) {
            hcsparseDestroySpVec(vec_x);
        }
    }
};

static hpccDataType dataTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        return HPCC_R_16F;
    case INFINI_DTYPE_F32:
        return HPCC_R_32F;
    case INFINI_DTYPE_F64:
        return HPCC_R_64F;
    default:
        return HPCC_R_32F;
    }
}

static bool isAligned16(const void *ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xf) == 0;
}

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
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    CHECK_OR_RETURN(pattern_desc->indicesDesc()->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(isAligned16(pattern_desc->indices()), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(isAligned16(output), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto result = SparseGatherInfo::create(output_desc, pattern_desc, input_desc);
    CHECK_RESULT(result);
    CHECK_OR_RETURN(result->input_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(result->output_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto opaque = new Opaque(handle->internal());
    auto info = result.take();
    auto data_type = dataTypeOf(dtype);

    auto status = hcsparseCreateSpVec(
        &opaque->vec_x,
        static_cast<int64_t>(info.input_vector.size),
        static_cast<int64_t>(info.nnz),
        const_cast<void *>(pattern_desc->indices()),
        output,
        HCSPARSE_INDEX_32I,
        HCSPARSE_INDEX_BASE_ZERO,
        data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = hcsparseCreateDnVec(
        &opaque->vec_y,
        static_cast<int64_t>(info.input_vector.size),
        const_cast<void *>(input),
        data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    *desc_ptr = new Descriptor(
        dtype,
        pattern_desc->indicesDesc()->dtype(),
        std::move(info),
        pattern_desc,
        0,
        opaque,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *,
    size_t,
    void *,
    const void *,
    void *stream) const {
    return _opaque->internal->useMcsparse(
        reinterpret_cast<hcStream_t>(stream),
        [&](hcsparseHandle_t sparse_handle) {
            CHECK_MCSPARSE(hcsparseGather(
                sparse_handle,
                _opaque->vec_y,
                _opaque->vec_x));
            return INFINI_STATUS_SUCCESS;
        });
}

} // namespace op::sparse_gather::metax
