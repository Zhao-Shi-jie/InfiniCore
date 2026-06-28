#ifndef __SPARSE_GATHER_INFO_H__
#define __SPARSE_GATHER_INFO_H__

#include "../../../utils.h"
#include "../../../utils/result.hpp"
#include "../../operator.h"
#include "../../spvec.h"
#include "../../tensor.h"

namespace op::sparse_gather {

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

class SparseGatherInfo {
    SparseGatherInfo() = default;

public:
    size_t nnz;
    DenseVector input_vector;
    ptrdiff_t output_stride;

    static utils::Result<SparseGatherInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopSpVecDescriptor_t pattern_desc,
        infiniopTensorDescriptor_t input_desc) {
        CHECK_OR_RETURN(pattern_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(output_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->dim(0) == pattern_desc->nnz(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->stride(0) != 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_DTYPE(pattern_desc->indicesDesc()->dtype(), INFINI_DTYPE_I32, INFINI_DTYPE_I64);

        auto input_vector = DenseVector::create(input_desc);
        CHECK_RESULT(input_vector);
        CHECK_OR_RETURN(input_vector->size == pattern_desc->size(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto dtype = output_desc->dtype();
        CHECK_OR_RETURN(input_desc->dtype() == dtype && pattern_desc->valuesDesc()->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<SparseGatherInfo>(SparseGatherInfo{
            pattern_desc->nnz(),
            input_vector.take(),
            output_desc->stride(0)});
    }
};

} // namespace op::sparse_gather

#endif // __SPARSE_GATHER_INFO_H__
