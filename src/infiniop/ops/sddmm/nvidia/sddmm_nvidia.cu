#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "sddmm_nvidia.cuh"

#include <cusparse.h>

namespace op::sddmm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    cusparseSpMatDescr_t mat_c = nullptr;
    cusparseDnMatDescr_t mat_a = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseOperation_t op_a = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t op_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSDDMMAlg_t alg = CUSPARSE_SDDMM_ALG_DEFAULT;
    cudaDataType data_type = CUDA_R_32F;
    cusparseIndexType_t index_type = CUSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (mat_c != nullptr) {
            cusparseDestroySpMat(mat_c);
        }
        if (mat_a != nullptr) {
            cusparseDestroyDnMat(mat_a);
        }
        if (mat_b != nullptr) {
            cusparseDestroyDnMat(mat_b);
        }
    }
};

static cudaDataType cudaDataTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
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
    infiniopSpMatDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->valuesDesc()->dtype();
    auto index_dtype = c_desc->crowIndicesDesc()->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32);

    auto result = SDDMMInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    auto a_layout = denseLayoutOf(info.a_matrix);
    CHECK_RESULT(a_layout);
    auto b_layout = denseLayoutOf(info.b_matrix);
    CHECK_RESULT(b_layout);

    auto opaque = new Opaque(handle->internal());
    opaque->data_type = cudaDataTypeOf(dtype);
    opaque->index_type = indexTypeOf(index_dtype);

    auto status = cusparseCreateCsr(
        &opaque->mat_c,
        static_cast<int64_t>(info.m),
        static_cast<int64_t>(info.n),
        static_cast<int64_t>(info.nnz),
        const_cast<void *>(c_desc->crowIndices()),
        const_cast<void *>(c_desc->colIndices()),
        const_cast<void *>(c_desc->values()),
        opaque->index_type,
        opaque->index_type,
        CUSPARSE_INDEX_BASE_ZERO,
        opaque->data_type);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = cusparseCreateDnMat(
        &opaque->mat_a,
        a_layout->rows,
        a_layout->cols,
        a_layout->ld,
        const_cast<void *>(c_desc->values()),
        opaque->data_type,
        a_layout->order);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = cusparseCreateDnMat(
        &opaque->mat_b,
        b_layout->rows,
        b_layout->cols,
        b_layout->ld,
        const_cast<void *>(c_desc->values()),
        opaque->data_type,
        b_layout->order);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    size_t workspace_size = 0;
    float alpha_one = 1.0f;
    float beta_zero = 0.0f;
    auto buffer_status = opaque->internal->useCusparse(nullptr, [&](cusparseHandle_t sparse_handle) {
        CHECK_CUSPARSE(cusparseSDDMM_bufferSize(
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
        c_desc,
        workspace_size,
        opaque,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c_values,
    const void *a,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_CUSPARSE(cusparseDnMatSetValues(_opaque->mat_a, const_cast<void *>(a)));
    CHECK_CUSPARSE(cusparseDnMatSetValues(_opaque->mat_b, const_cast<void *>(b)));
    CHECK_CUSPARSE(cusparseSpMatSetValues(_opaque->mat_c, c_values));

    CHECK_STATUS(_opaque->internal->useCusparse(
        reinterpret_cast<cudaStream_t>(stream),
        [&](cusparseHandle_t sparse_handle) {
            CHECK_CUSPARSE(cusparseSDDMM(
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

} // namespace op::sddmm::nvidia
