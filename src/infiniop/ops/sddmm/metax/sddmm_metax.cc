#include "sddmm_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace op::sddmm::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    hcsparseSpMatDescr_t mat_c = nullptr;
    hcsparseDnMatDescr_t mat_a = nullptr;
    hcsparseDnMatDescr_t mat_b = nullptr;
    int32_t *crow_i32 = nullptr;
    int32_t *col_i32 = nullptr;
    void *crow_indices = nullptr;
    void *col_indices = nullptr;
    hcsparseOperation_t op_a = HCSPARSE_OPERATION_NON_TRANSPOSE;
    hcsparseOperation_t op_b = HCSPARSE_OPERATION_NON_TRANSPOSE;
    hcsparseSDDMMAlg_t alg = HCSPARSE_SDDMM_ALG_DEFAULT;
    hpccDataType data_type = HPCC_R_32F;
    hcsparseIndexType_t index_type = HCSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::metax::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (mat_c != nullptr) {
            hcsparseDestroySpMat(mat_c);
        }
        if (mat_a != nullptr) {
            hcsparseDestroyDnMat(mat_a);
        }
        if (mat_b != nullptr) {
            hcsparseDestroyDnMat(mat_b);
        }
        if (crow_i32 != nullptr) {
            hcFree(crow_i32);
        }
        if (col_i32 != nullptr) {
            hcFree(col_i32);
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
    hcsparseOperation_t op;
};

static utils::Result<DenseLayout> denseLayoutOf(const DenseMatrix &matrix) {
    if (matrix.col_stride == 1) {
        CHECK_OR_RETURN(matrix.row_stride > 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseLayout>(DenseLayout{
            static_cast<int64_t>(matrix.cols),
            static_cast<int64_t>(matrix.rows),
            static_cast<int64_t>(matrix.row_stride),
            HCSPARSE_ORDER_COL,
            HCSPARSE_OPERATION_TRANSPOSE});
    }

    if (matrix.row_stride == 1) {
        CHECK_OR_RETURN(matrix.col_stride > 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseLayout>(DenseLayout{
            static_cast<int64_t>(matrix.rows),
            static_cast<int64_t>(matrix.cols),
            static_cast<int64_t>(matrix.col_stride),
            HCSPARSE_ORDER_COL,
            HCSPARSE_OPERATION_NON_TRANSPOSE});
    }

    return INFINI_STATUS_BAD_TENSOR_STRIDES;
}

static bool fitsI32(int64_t value) {
    return value >= 0 && value <= std::numeric_limits<int32_t>::max();
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
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = c_desc->valuesDesc()->dtype();
    auto index_dtype = c_desc->crowIndicesDesc()->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SDDMMInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    auto a_layout = denseLayoutOf(info.a_matrix);
    CHECK_RESULT(a_layout);
    auto b_layout = denseLayoutOf(info.b_matrix);
    CHECK_RESULT(b_layout);

    auto opaque = new Opaque(handle->internal());
    opaque->data_type = dataTypeOf(dtype);
    opaque->index_type = indexTypeOf(index_dtype);
    opaque->op_a = a_layout->op;
    opaque->op_b = b_layout->op;

    auto *crow_indices = const_cast<void *>(c_desc->crowIndices());
    auto *col_indices = const_cast<void *>(c_desc->colIndices());
    if (index_dtype == INFINI_DTYPE_I64) {
        std::vector<int64_t> crow_i64(info.m + 1);
        std::vector<int64_t> col_i64(info.nnz);
        CHECK_API_OR(
            hcMemcpy(crow_i64.data(), c_desc->crowIndices(), crow_i64.size() * sizeof(int64_t), hcMemcpyDeviceToHost),
            hcSuccess,
            {
                delete opaque;
                return INFINI_STATUS_INTERNAL_ERROR;
            });
        if (!col_i64.empty()) {
            CHECK_API_OR(
                hcMemcpy(col_i64.data(), c_desc->colIndices(), col_i64.size() * sizeof(int64_t), hcMemcpyDeviceToHost),
                hcSuccess,
                {
                    delete opaque;
                    return INFINI_STATUS_INTERNAL_ERROR;
                });
        }

        std::vector<int32_t> crow_i32(crow_i64.size());
        std::vector<int32_t> col_i32(col_i64.size());
        for (size_t i = 0; i < crow_i64.size(); ++i) {
            CHECK_OR_DO(fitsI32(crow_i64[i]), {
                delete opaque;
                return INFINI_STATUS_BAD_PARAM;
            });
            crow_i32[i] = static_cast<int32_t>(crow_i64[i]);
        }
        for (size_t i = 0; i < col_i64.size(); ++i) {
            CHECK_OR_DO(fitsI32(col_i64[i]), {
                delete opaque;
                return INFINI_STATUS_BAD_PARAM;
            });
            col_i32[i] = static_cast<int32_t>(col_i64[i]);
        }

        CHECK_API_OR(
            hcMalloc(reinterpret_cast<void **>(&opaque->crow_i32), crow_i32.size() * sizeof(int32_t)),
            hcSuccess,
            {
                delete opaque;
                return INFINI_STATUS_INTERNAL_ERROR;
            });
        CHECK_API_OR(
            hcMemcpy(opaque->crow_i32, crow_i32.data(), crow_i32.size() * sizeof(int32_t), hcMemcpyHostToDevice),
            hcSuccess,
            {
                delete opaque;
                return INFINI_STATUS_INTERNAL_ERROR;
            });

        if (!col_i32.empty()) {
            CHECK_API_OR(
                hcMalloc(reinterpret_cast<void **>(&opaque->col_i32), col_i32.size() * sizeof(int32_t)),
                hcSuccess,
                {
                    delete opaque;
                    return INFINI_STATUS_INTERNAL_ERROR;
                });
            CHECK_API_OR(
                hcMemcpy(opaque->col_i32, col_i32.data(), col_i32.size() * sizeof(int32_t), hcMemcpyHostToDevice),
                hcSuccess,
                {
                    delete opaque;
                    return INFINI_STATUS_INTERNAL_ERROR;
                });
        }

        crow_indices = opaque->crow_i32;
        col_indices = opaque->col_i32;
        opaque->index_type = HCSPARSE_INDEX_32I;
    }
    opaque->crow_indices = crow_indices;
    opaque->col_indices = col_indices;

    auto status = hcsparseCreateCsr(
        &opaque->mat_c,
        static_cast<int64_t>(info.m),
        static_cast<int64_t>(info.n),
        static_cast<int64_t>(info.nnz),
        crow_indices,
        col_indices,
        const_cast<void *>(c_desc->values()),
        opaque->index_type,
        opaque->index_type,
        HCSPARSE_INDEX_BASE_ZERO,
        opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = hcsparseCreateDnMat(
        &opaque->mat_a,
        a_layout->rows,
        a_layout->cols,
        a_layout->ld,
        const_cast<void *>(c_desc->values()),
        opaque->data_type,
        a_layout->order);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = hcsparseCreateDnMat(
        &opaque->mat_b,
        b_layout->rows,
        b_layout->cols,
        b_layout->ld,
        const_cast<void *>(c_desc->values()),
        opaque->data_type,
        b_layout->order);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    size_t workspace_size = 0;
    float alpha_one = 1.0f;
    float beta_zero = 0.0f;
    auto buffer_status = opaque->internal->useMcsparse(nullptr, [&](hcsparseHandle_t sparse_handle) {
        CHECK_MCSPARSE(hcsparseSDDMM_bufferSize(
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

    CHECK_MCSPARSE(hcsparseDnMatSetValues(_opaque->mat_a, const_cast<void *>(a)));
    CHECK_MCSPARSE(hcsparseDnMatSetValues(_opaque->mat_b, const_cast<void *>(b)));
    CHECK_MCSPARSE(hcsparseCsrSetPointers(
        _opaque->mat_c,
        _opaque->crow_indices,
        _opaque->col_indices,
        c_values));

    CHECK_STATUS(_opaque->internal->useMcsparse(
        reinterpret_cast<hcStream_t>(stream),
        [&](hcsparseHandle_t sparse_handle) {
            CHECK_MCSPARSE(hcsparseSDDMM(
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

} // namespace op::sddmm::metax
