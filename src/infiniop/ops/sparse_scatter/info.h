#ifndef __SPARSE_SCATTER_INFO_H__
#define __SPARSE_SCATTER_INFO_H__

#include "../../../utils.h"
#include "../../../utils/result.hpp"
#include "../../operator.h"
#include "../../spvec.h"
#include "../../tensor.h"

namespace op::sparse_scatter {

struct DenseVector {
    size_t size;
    ptrdiff_t stride;

    static utils::Result<DenseVector> create(infiniopTensorDescriptor_t desc) {
        CHECK_OR_RETURN(desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(desc->stride(0) != 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
        return utils::Result<DenseVector>(DenseVector{desc->dim(0), desc->stride(0)});
    }
};

class SparseScatterInfo {
    SparseScatterInfo() = default;

public:
    size_t nnz;
    DenseVector output_vector;

    static utils::Result<SparseScatterInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopSpVecDescriptor_t input_desc) {
        CHECK_OR_RETURN(input_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_DTYPE(input_desc->indicesDesc()->dtype(), INFINI_DTYPE_I32, INFINI_DTYPE_I64);

        auto output_vector = DenseVector::create(output_desc);
        CHECK_RESULT(output_vector);
        CHECK_OR_RETURN(output_vector->size == input_desc->size(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto dtype = output_desc->dtype();
        CHECK_OR_RETURN(input_desc->valuesDesc()->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<SparseScatterInfo>(SparseScatterInfo{
            input_desc->nnz(),
            output_vector.take()});
    }
};

} // namespace op::sparse_scatter

#endif // __SPARSE_SCATTER_INFO_H__
