#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "spmm_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cusparse.h>

namespace op::spmm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseDnMatDescr_t mat_c = nullptr;
    cusparseOperation_t op_a = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t op_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;
    cudaDataType data_type = CUDA_R_32F;
    cusparseIndexType_t index_type = CUSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (mat_a != nullptr) {
            cusparseDestroySpMat(mat_a);
        }
        if (mat_b != nullptr) {
            cusparseDestroyDnMat(mat_b);
        }
        if (mat_c != nullptr) {
            cusparseDestroyDnMat(mat_c);
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

struct DenseLayout {
    int64_t rows;
    int64_t cols;
    int64_t ld;
    cusparseOrder_t order;
};

static utils::Result<DenseLayout> denseLayoutOf(const DenseMatrix &matrix) {
    if (matrix.col_stride == 1) {
        CHECK_OR_RETURN(matrix.row_stride > 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseLayout>(DenseLayout{
            static_cast<int64_t>(matrix.rows),
            static_cast<int64_t>(matrix.cols),
            static_cast<int64_t>(matrix.row_stride),
            CUSPARSE_ORDER_ROW});
    }

    if (matrix.row_stride == 1) {
        CHECK_OR_RETURN(matrix.col_stride > 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseLayout>(DenseLayout{
            static_cast<int64_t>(matrix.rows),
            static_cast<int64_t>(matrix.cols),
            static_cast<int64_t>(matrix.col_stride),
            CUSPARSE_ORDER_COL});
    }

    return INFINI_STATUS_BAD_TENSOR_STRIDES;
}

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();
    auto index_dtype = a_desc->crowIndicesDesc()->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SpMMInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    auto b_layout = denseLayoutOf(info.b_matrix);
    CHECK_RESULT(b_layout);
    auto c_layout = denseLayoutOf(info.c_matrix);
    CHECK_RESULT(c_layout);

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

    status = cusparseCreateDnMat(
        &opaque->mat_b,
        b_layout->rows,
        b_layout->cols,
        b_layout->ld,
        const_cast<void *>(a_desc->values()),
        opaque->data_type,
        b_layout->order);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = cusparseCreateDnMat(
        &opaque->mat_c,
        c_layout->rows,
        c_layout->cols,
        c_layout->ld,
        const_cast<void *>(a_desc->values()),
        opaque->data_type,
        c_layout->order);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    size_t workspace_size = 0;
    float alpha_one = 1.0f;
    float beta_zero = 0.0f;
    auto buffer_status = opaque->internal->useCusparse(nullptr, [&](cusparseHandle_t sparse_handle) {
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            sparse_handle,
            opaque->op_a,
            opaque->op_b,
            &alpha_one,
            opaque->mat_a,
            opaque->mat_b,
            &beta_zero,
            opaque->mat_c,
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
    void *c,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_CUSPARSE(cusparseDnMatSetValues(_opaque->mat_b, const_cast<void *>(b)));
    CHECK_CUSPARSE(cusparseDnMatSetValues(_opaque->mat_c, c));

    CHECK_STATUS(_opaque->internal->useCusparse(
        reinterpret_cast<cudaStream_t>(stream),
        [&](cusparseHandle_t sparse_handle) {
            CHECK_CUSPARSE(cusparseSpMM(
                sparse_handle,
                _opaque->op_a,
                _opaque->op_b,
                &alpha,
                _opaque->mat_a,
                _opaque->mat_b,
                &beta,
                _opaque->mat_c,
                CUDA_R_32F,
                _opaque->alg,
                workspace));
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmm::nvidia
