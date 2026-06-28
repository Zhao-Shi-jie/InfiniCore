#ifndef __SPVV_INFO_H__
#define __SPVV_INFO_H__

#include "../../../utils.h"
#include "../../../utils/result.hpp"
#include "../../operator.h"
#include "../../spvec.h"
#include "../../tensor.h"

namespace op::spvv {

struct DenseScalar {
    static utils::Result<DenseScalar> create(infiniopTensorDescriptor_t desc) {
        auto shape = desc->shape();
        CHECK_OR_RETURN(shape.size() == 0 || (shape.size() == 1 && shape[0] == 1), INFINI_STATUS_BAD_TENSOR_SHAPE);
        return utils::Result<DenseScalar>(DenseScalar{});
    }
};

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

class SpVVInfo {
    SpVVInfo() = default;

public:
    size_t size;
    size_t nnz;
    DenseVector x_vector;

    static utils::Result<SpVVInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopSpVecDescriptor_t a_desc,
        infiniopTensorDescriptor_t x_desc) {

        CHECK_OR_RETURN(a_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto y_scalar = DenseScalar::create(y_desc);
        CHECK_RESULT(y_scalar);

        auto x_vector = DenseVector::create(x_desc);
        CHECK_RESULT(x_vector);

        CHECK_OR_RETURN(x_vector->size == a_desc->size(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto dtype = y_desc->dtype();
        CHECK_OR_RETURN(x_desc->dtype() == dtype && a_desc->valuesDesc()->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<SpVVInfo>(SpVVInfo{
            a_desc->size(),
            a_desc->nnz(),
            x_vector.take()});
    }
};

} // namespace op::spvv

#endif // __SPVV_INFO_H__
