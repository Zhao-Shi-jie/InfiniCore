#include "spmm_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::spmm::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    hcsparseSpMatDescr_t mat_a = nullptr;
    hcsparseDnMatDescr_t mat_b = nullptr;
    hcsparseDnMatDescr_t mat_c = nullptr;
    hcsparseOperation_t op_a = HCSPARSE_OPERATION_NON_TRANSPOSE;
    hcsparseOperation_t op_b = HCSPARSE_OPERATION_NON_TRANSPOSE;
    hcsparseSpMMAlg_t alg = HCSPARSE_SPMM_ALG_DEFAULT;
    hpccDataType data_type = HPCC_R_32F;
    hcsparseIndexType_t index_type = HCSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::metax::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (mat_a != nullptr) {
            hcsparseDestroySpMat(mat_a);
        }
        if (mat_b != nullptr) {
            hcsparseDestroyDnMat(mat_b);
        }
        if (mat_c != nullptr) {
            hcsparseDestroyDnMat(mat_c);
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

struct DenseLayout {
    int64_t rows;
    int64_t cols;
    int64_t ld;
    hcsparseOrder_t order;
};

static utils::Result<DenseLayout> denseLayoutOf(const DenseMatrix &matrix) {
    if (matrix.col_stride == 1) {
        CHECK_OR_RETURN(matrix.row_stride > 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseLayout>(DenseLayout{
            static_cast<int64_t>(matrix.rows),
            static_cast<int64_t>(matrix.cols),
            static_cast<int64_t>(matrix.row_stride),
            HCSPARSE_ORDER_ROW});
    }

    if (matrix.row_stride == 1) {
        CHECK_OR_RETURN(matrix.col_stride > 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseLayout>(DenseLayout{
            static_cast<int64_t>(matrix.rows),
            static_cast<int64_t>(matrix.cols),
            static_cast<int64_t>(matrix.col_stride),
            HCSPARSE_ORDER_COL});
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
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
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

    status = hcsparseCreateDnMat(
        &opaque->mat_b,
        b_layout->rows,
        b_layout->cols,
        b_layout->ld,
        const_cast<void *>(a_desc->values()),
        opaque->data_type,
        b_layout->order);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = hcsparseCreateDnMat(
        &opaque->mat_c,
        c_layout->rows,
        c_layout->cols,
        c_layout->ld,
        const_cast<void *>(a_desc->values()),
        opaque->data_type,
        c_layout->order);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    size_t workspace_size = 0;
    float alpha_one = 1.0f;
    float beta_zero = 0.0f;
    auto buffer_status = opaque->internal->useMcsparse(nullptr, [&](hcsparseHandle_t sparse_handle) {
        CHECK_MCSPARSE(hcsparseSpMM_bufferSize(
            sparse_handle,
            opaque->op_a,
            opaque->op_b,
            &alpha_one,
            opaque->mat_a,
            opaque->mat_b,
            &beta_zero,
            opaque->mat_c,
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
    void *c,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_MCSPARSE(hcsparseDnMatSetValues(_opaque->mat_b, const_cast<void *>(b)));
    CHECK_MCSPARSE(hcsparseDnMatSetValues(_opaque->mat_c, c));

    CHECK_STATUS(_opaque->internal->useMcsparse(
        reinterpret_cast<hcStream_t>(stream),
        [&](hcsparseHandle_t sparse_handle) {
            CHECK_MCSPARSE(hcsparseSpMM(
                sparse_handle,
                _opaque->op_a,
                _opaque->op_b,
                &alpha,
                _opaque->mat_a,
                _opaque->mat_b,
                &beta,
                _opaque->mat_c,
                HPCC_R_32F,
                _opaque->alg,
                workspace));
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmm::metax
