#ifndef __SPMM_INFO_H__
#define __SPMM_INFO_H__

#include "../../../utils.h"
#include "../../../utils/result.hpp"
#include "../../operator.h"
#include "../../spmat.h"
#include "../../tensor.h"

namespace op::spmm {

struct DenseMatrix {
    size_t rows;
    size_t cols;
    ptrdiff_t row_stride;
    ptrdiff_t col_stride;

    static utils::Result<DenseMatrix> create(infiniopTensorDescriptor_t desc) {
        CHECK_OR_RETURN(desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(desc->stride(0) != 0 && desc->stride(1) != 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseMatrix>(DenseMatrix{
            desc->dim(0),
            desc->dim(1),
            desc->stride(0),
            desc->stride(1)});
    }
};

class SpMMInfo {
    SpMMInfo() = default;

public:
    size_t m;
    size_t n;
    size_t k;
    size_t nnz;
    DenseMatrix b_matrix;
    DenseMatrix c_matrix;

    static utils::Result<SpMMInfo> create(
        infiniopTensorDescriptor_t c_desc,
        infiniopSpMatDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc) {

        CHECK_OR_RETURN(a_desc->format() == INFINIOP_SPMAT_FORMAT_CSR, INFINI_STATUS_BAD_PARAM);

        auto b_matrix = DenseMatrix::create(b_desc);
        CHECK_RESULT(b_matrix);

        auto c_matrix = DenseMatrix::create(c_desc);
        CHECK_RESULT(c_matrix);

        CHECK_OR_RETURN(c_matrix->rows == a_desc->rows(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(b_matrix->rows == a_desc->cols(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(c_matrix->cols == b_matrix->cols, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto dtype = c_desc->dtype();
        CHECK_OR_RETURN(b_desc->dtype() == dtype && a_desc->valuesDesc()->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<SpMMInfo>(SpMMInfo{
            a_desc->rows(),
            b_matrix->cols,
            a_desc->cols(),
            a_desc->nnz(),
            b_matrix.take(),
            c_matrix.take()});
    }
};

} // namespace op::spmm

#endif // __SPMM_INFO_H__
