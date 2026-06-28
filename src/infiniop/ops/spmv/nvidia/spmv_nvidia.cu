#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "spmv_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cusparse.h>

namespace op::spmv::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    cusparseOperation_t op_a = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;
    cudaDataType data_type = CUDA_R_32F;
    cusparseIndexType_t index_type = CUSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (mat_a != nullptr) {
            cusparseDestroySpMat(mat_a);
        }
        if (vec_x != nullptr) {
            cusparseDestroyDnVec(vec_x);
        }
        if (vec_y != nullptr) {
            cusparseDestroyDnVec(vec_y);
        }
    }
};

static cudaDataType cudaDataTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        return CUDA_R_16F;
    case INFINI_DTYPE_BF16:
        return CUDA_R_16BF;
    case INFINI_DTYPE_F32:
        return CUDA_R_32F;
    default:
        return CUDA_R_32F;
    }
}

static cusparseIndexType_t indexTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_I32:
        return CUSPARSE_INDEX_32I;
    case INFINI_DTYPE_I64:
        return CUSPARSE_INDEX_64I;
    default:
        return CUSPARSE_INDEX_64I;
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
    (void)y;
    (void)x;
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = y_desc->dtype();
    auto index_dtype = a_desc->crowIndicesDesc()->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SpMVInfo::create(y_desc, a_desc, x_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    CHECK_OR_RETURN(info.x_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(info.y_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto opaque = new Opaque(handle->internal());
    opaque->data_type = cudaDataTypeOf(dtype);
    opaque->index_type = indexTypeOf(index_dtype);

    auto status = cusparseCreateCsr(
        &opaque->mat_a,
        static_cast<int64_t>(info.m),
        static_cast<int64_t>(info.k),
        static_cast<int64_t>(info.nnz),
        const_cast<void *>(a_desc->crowIndices()),
        const_cast<void *>(a_desc->colIndices()),
        const_cast<void *>(a_desc->values()),
        opaque->index_type,
        opaque->index_type,
        CUSPARSE_INDEX_BASE_ZERO,
        opaque->data_type);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = cusparseCreateDnVec(
        &opaque->vec_x,
        static_cast<int64_t>(info.k),
        const_cast<void *>(a_desc->values()),
        opaque->data_type);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = cusparseCreateDnVec(
        &opaque->vec_y,
        static_cast<int64_t>(info.m),
        const_cast<void *>(a_desc->values()),
        opaque->data_type);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    size_t workspace_size = 0;
    float alpha_one = 1.0f;
    float beta_zero = 0.0f;
    auto buffer_status = opaque->internal->useCusparse(nullptr, [&](cusparseHandle_t sparse_handle) {
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            sparse_handle,
            opaque->op_a,
            &alpha_one,
            opaque->mat_a,
            opaque->vec_x,
            &beta_zero,
            opaque->vec_y,
            CUDA_R_32F,
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
    void *y,
    const void *x,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_CUSPARSE(cusparseDnVecSetValues(_opaque->vec_x, const_cast<void *>(x)));
    CHECK_CUSPARSE(cusparseDnVecSetValues(_opaque->vec_y, y));

    CHECK_STATUS(_opaque->internal->useCusparse(
        reinterpret_cast<cudaStream_t>(stream),
        [&](cusparseHandle_t sparse_handle) {
            CHECK_CUSPARSE(cusparseSpMV(
                sparse_handle,
                _opaque->op_a,
                &alpha,
                _opaque->mat_a,
                _opaque->vec_x,
                &beta,
                _opaque->vec_y,
                CUDA_R_32F,
                _opaque->alg,
                workspace));
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv::nvidia
