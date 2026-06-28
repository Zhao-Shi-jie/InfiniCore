#ifndef __SPMV_INFO_H__
#define __SPMV_INFO_H__

#include "../../../utils.h"
#include "../../../utils/result.hpp"
#include "../../operator.h"
#include "../../spmat.h"
#include "../../tensor.h"

namespace op::spmv {

struct DenseVector {
    size_t size;
    ptrdiff_t stride;

    static utils::Result<DenseVector> create(infiniopTensorDescriptor_t desc) {
        CHECK_OR_RETURN(desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(desc->stride(0) != 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseVector>(DenseVector{
            desc->dim(0),
            desc->stride(0)});
    }
};

class SpMVInfo {
    SpMVInfo() = default;

public:
    size_t m;
    size_t k;
    size_t nnz;
    DenseVector x_vector;
    DenseVector y_vector;

    static utils::Result<SpMVInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopSpMatDescriptor_t a_desc,
        infiniopTensorDescriptor_t x_desc) {

        CHECK_OR_RETURN(a_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(a_desc->format() == INFINIOP_SPMAT_FORMAT_CSR, INFINI_STATUS_BAD_PARAM);

        auto x_vector = DenseVector::create(x_desc);
        CHECK_RESULT(x_vector);

        auto y_vector = DenseVector::create(y_desc);
        CHECK_RESULT(y_vector);

        CHECK_OR_RETURN(y_vector->size == a_desc->rows(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_vector->size == a_desc->cols(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto dtype = y_desc->dtype();
        CHECK_OR_RETURN(x_desc->dtype() == dtype && a_desc->valuesDesc()->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<SpMVInfo>(SpMVInfo{
            a_desc->rows(),
            a_desc->cols(),
            a_desc->nnz(),
            x_vector.take(),
            y_vector.take()});
    }
};

} // namespace op::spmv

#endif // __SPMV_INFO_H__
