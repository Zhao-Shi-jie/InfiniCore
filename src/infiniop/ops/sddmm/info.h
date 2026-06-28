#ifndef __SDDMM_INFO_H__
#define __SDDMM_INFO_H__

#include "../../../utils.h"
#include "../../../utils/result.hpp"
#include "../../operator.h"
#include "../../spmat.h"
#include "../../tensor.h"

namespace op::sddmm {

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

class SDDMMInfo {
    SDDMMInfo() = default;

public:
    size_t m;
    size_t n;
    size_t k;
    size_t nnz;
    DenseMatrix a_matrix;
    DenseMatrix b_matrix;

    static utils::Result<SDDMMInfo> create(
        infiniopSpMatDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc) {
        CHECK_OR_RETURN(c_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(c_desc->format() == INFINIOP_SPMAT_FORMAT_CSR, INFINI_STATUS_BAD_PARAM);

        auto a_matrix = DenseMatrix::create(a_desc);
        CHECK_RESULT(a_matrix);
        auto b_matrix = DenseMatrix::create(b_desc);
        CHECK_RESULT(b_matrix);

        CHECK_OR_RETURN(a_matrix->rows == c_desc->rows(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(b_matrix->cols == c_desc->cols(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(a_matrix->cols == b_matrix->rows, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto dtype = c_desc->valuesDesc()->dtype();
        CHECK_OR_RETURN(a_desc->dtype() == dtype && b_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<SDDMMInfo>(SDDMMInfo{
            c_desc->rows(),
            c_desc->cols(),
            a_matrix->cols,
            c_desc->nnz(),
            a_matrix.take(),
            b_matrix.take()});
    }
};

} // namespace op::sddmm

#endif // __SDDMM_INFO_H__
