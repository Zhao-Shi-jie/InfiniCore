#include "spmv_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::spmv::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    hcsparseSpMatDescr_t mat_a = nullptr;
    hcsparseDnVecDescr_t vec_x = nullptr;
    hcsparseDnVecDescr_t vec_y = nullptr;
    hcsparseOperation_t op_a = HCSPARSE_OPERATION_NON_TRANSPOSE;
    hcsparseSpMVAlg_t alg = HCSPARSE_SPMV_ALG_DEFAULT;
    hpccDataType data_type = HPCC_R_32F;
    hcsparseIndexType_t index_type = HCSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::metax::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (mat_a != nullptr) {
            hcsparseDestroySpMat(mat_a);
        }
        if (vec_x != nullptr) {
            hcsparseDestroyDnVec(vec_x);
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
    infiniopTensorDescriptor_t y_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t x_desc,
    void *y,
    const void *x) {
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = y_desc->dtype();
    auto index_dtype = a_desc->crowIndicesDesc()->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SpMVInfo::create(y_desc, a_desc, x_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    CHECK_OR_RETURN(info.x_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(info.y_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto opaque = new Opaque(handle->internal());
    opaque->data_type = dataTypeOf(dtype);
    opaque->index_type = indexTypeOf(index_dtype);

    auto status = hcsparseCreateCsr(
        &opaque->mat_a,
        static_cast<int64_t>(info.m),
        static_cast<int64_t>(info.k),
        static_cast<int64_t>(info.nnz),
        const_cast<void *>(a_desc->crowIndices()),
        const_cast<void *>(a_desc->colIndices()),
        const_cast<void *>(a_desc->values()),
        opaque->index_type,
        opaque->index_type,
        HCSPARSE_INDEX_BASE_ZERO,
        opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = hcsparseCreateDnVec(
        &opaque->vec_x,
        static_cast<int64_t>(info.k),
        const_cast<void *>(x),
        opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = hcsparseCreateDnVec(
        &opaque->vec_y,
        static_cast<int64_t>(info.m),
        y,
        opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    size_t workspace_size = 0;
    float alpha_one = 1.0f;
    float beta_zero = 0.0f;
    auto buffer_status = opaque->internal->useMcsparse(nullptr, [&](hcsparseHandle_t sparse_handle) {
        CHECK_MCSPARSE(hcsparseSpMV_bufferSize(
            sparse_handle,
            opaque->op_a,
            &alpha_one,
            opaque->mat_a,
            opaque->vec_x,
            &beta_zero,
            opaque->vec_y,
            HPCC_R_32F,
            opaque->alg,
            &workspace_size));
        return INFINI_STATUS_SUCCESS;
    });
    CHECK_API_OR(buffer_status, INFINI_STATUS_SUCCESS, {
        delete opaque;
        return buffer_status;
    });

    *desc_ptr = new Descriptor(
        dtype,
        index_dtype,
        info,
        a_desc,
        workspace_size,
        opaque,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void * /*y*/,
    const void * /*x*/,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_STATUS(_opaque->internal->useMcsparse(
        reinterpret_cast<hcStream_t>(stream),
        [&](hcsparseHandle_t sparse_handle) {
            CHECK_MCSPARSE(hcsparseSpMV(
                sparse_handle,
                _opaque->op_a,
                &alpha,
                _opaque->mat_a,
                _opaque->vec_x,
                &beta,
                _opaque->vec_y,
                HPCC_R_32F,
                _opaque->alg,
                workspace));
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv::metax
