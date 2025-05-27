#ifndef __SPMV_INFO_H__
#define __SPMV_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::spmv {

enum class SparseFormat {
    CSR = 0,
    COO = 1
};

struct SpMVInfo {
    size_t num_rows;     // Number of rows in the matrix
    size_t num_cols;     // Number of columns in the matrix
    size_t nnz;          // Number of non-zero elements
    SparseFormat format; // Format of the sparse matrix
    infiniDtype_t dtype; // Data type of the values

    static utils::Result<SpMVInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t values_desc,
        infiniopTensorDescriptor_t row_indices_desc,
        infiniopTensorDescriptor_t col_indices_desc,
        SparseFormat format) {

        SpMVInfo info;

        // Check that the data types are compatible
        auto dtype_values = values_desc->dtype();
        auto dtype_x = x_desc->dtype();
        auto dtype_y = y_desc->dtype();

        if (dtype_values != dtype_x || dtype_values != dtype_y) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        info.dtype = dtype_values;

        // Check that the shapes are valid
        auto y_shape = y_desc->shape();
        auto x_shape = x_desc->shape();
        auto values_shape = values_desc->shape();
        auto row_indices_shape = row_indices_desc->shape();
        auto col_indices_shape = col_indices_desc->shape();

        // Verify that values, row_indices, and col_indices have the same number of elements
        if (format == SparseFormat::CSR) {
            // For CSR, row_indices should have num_rows + 1 elements
            if (row_indices_shape.size() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            info.num_rows = row_indices_shape[0] - 1;

            if (y_shape.size() != 1 || y_shape[0] != info.num_rows) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else if (format == SparseFormat::COO) {
            // For COO, row_indices should have nnz elements
            if (values_shape.size() != 1 || row_indices_shape.size() != 1 || col_indices_shape.size() != 1 || values_shape[0] != row_indices_shape[0] || values_shape[0] != col_indices_shape[0]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            info.nnz = values_shape[0];

            // For COO format, we need to determine num_rows from y_shape
            if (y_shape.size() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            info.num_rows = y_shape[0];
        } else {
            return INFINI_STATUS_BAD_PARAM;
        }

        // X shape should be 1D
        if (x_shape.size() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        info.num_cols = x_shape[0];

        // Check that the integer types for indices are correct (int32)
        if (row_indices_desc->dtype() != INFINI_DTYPE_INT32 || col_indices_desc->dtype() != INFINI_DTYPE_INT32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Set format
        info.format = format;

        return utils::Result<SpMVInfo>(info);
    }
};

} // namespace op::spmv

#endif // __SPMV_INFO_H__