#include "axpby_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::axpby::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    hcsparseSpVecDescr_t vec_x = nullptr;
    hcsparseDnVecDescr_t vec_y = nullptr;
    hpccDataType data_type = HPCC_R_32F;
    hcsparseIndexType_t index_type = HCSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::metax::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (vec_x != nullptr) {
            hcsparseDestroySpVec(vec_x);
        }
        if (vec_y != nullptr) {
            hcsparseDestroyDnVec(vec_y);
        }
    }
};

static hpccDataType dataTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        return HPCC_R_16F;
    case INFINI_DTYPE_BF16:
        return HPCC_R_16BF;
    case INFINI_DTYPE_F32:
        return HPCC_R_32F;
    case INFINI_DTYPE_F64:
        return HPCC_R_64F;
    default:
        return HPCC_R_32F;
    }
}

static hcsparseIndexType_t indexTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_I32:
        return HCSPARSE_INDEX_32I;
    case INFINI_DTYPE_I64:
        return HCSPARSE_INDEX_64I;
    default:
        return HCSPARSE_INDEX_64I;
    }
}

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopSpVecDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = AxpbyInfo::create(x_desc, y_desc);
    CHECK_RESULT(result);
    CHECK_OR_RETURN(result->incy == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
    auto opaque = new Opaque(handle->internal());
    opaque->data_type = dataTypeOf(y_desc->dtype());
    opaque->index_type = indexTypeOf(x_desc->indicesDesc()->dtype());

    auto status = hcsparseCreateSpVec(
        &opaque->vec_x,
        static_cast<int64_t>(result->n),
        static_cast<int64_t>(result->nnz),
        const_cast<void *>(x_desc->indices()),
        const_cast<void *>(x_desc->values()),
        opaque->index_type,
        HCSPARSE_INDEX_BASE_ZERO,
        opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = hcsparseCreateDnVec(
        &opaque->vec_y,
        static_cast<int64_t>(result->n),
        const_cast<void *>(x_desc->values()),
        opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    *desc_ptr = new Descriptor(
        result.take(),
        x_desc,
        0,
        opaque,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    float alpha,
    float beta,
    void *stream) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    CHECK_MCSPARSE(hcsparseDnVecSetValues(_opaque->vec_y, y));
    CHECK_STATUS(_opaque->internal->useMcsparse(
        reinterpret_cast<hcStream_t>(stream),
        [&](hcsparseHandle_t sparse_handle) {
            CHECK_MCSPARSE(hcsparseAxpby(
                sparse_handle,
                &alpha,
                _opaque->vec_x,
                &beta,
                _opaque->vec_y));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::axpby::metax
