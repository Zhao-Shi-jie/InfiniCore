#ifndef __INFINIOP_SPMM_INFO_H__
#define __INFINIOP_SPMM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::spmm {

struct SpmmInfo {
    size_t m;   // rows of sparse matrix (= rows of C)
    size_t k;   // cols of sparse matrix (= rows of B)
    size_t n;   // cols of B (= cols of C)
    size_t nnz; // number of non-zeros in the sparse matrix

    static utils::Result<SpmmInfo> create(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t values_desc,
        size_t rows,
        size_t cols) {

        if (c_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (b_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (values_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t m = rows;
        const size_t k = cols;
        const size_t n = b_desc->dim(1);
        const size_t nnz = values_desc->dim(0);

        if (c_desc->dim(0) != m || c_desc->dim(1) != n) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (b_desc->dim(0) != k) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (c_desc->dtype() != b_desc->dtype() || c_desc->dtype() != values_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return utils::Result<SpmmInfo>(SpmmInfo{m, k, n, nnz});
    }
};

} // namespace op::spmm

#endif // __INFINIOP_SPMM_INFO_H__
