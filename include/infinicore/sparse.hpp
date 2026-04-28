#pragma once

#include "tensor.hpp"

namespace infinicore {

/**
 * SpMat: Sparse Matrix in CSR (Compressed Sparse Row) format.
 *
 * The matrix is stored as three dense Tensors:
 *   - row_offsets: shape [rows+1], dtype I32 — row-pointer array
 *   - col_indices: shape [nnz],   dtype I32 — column-index array
 *   - values:      shape [nnz],   dtype T   — non-zero value array
 *
 * This follows the cuSPARSE CSR convention and intentionally re-uses the
 * existing Tensor primitive so that no new memory-management code is needed.
 */
class SpMat {
public:
    enum class Format { CSR };

    SpMat(Tensor row_offsets, Tensor col_indices, Tensor values,
          size_t rows, size_t cols)
        : row_offsets_(std::move(row_offsets)),
          col_indices_(std::move(col_indices)),
          values_(std::move(values)),
          rows_(rows),
          cols_(cols) {}

    const Tensor &row_offsets() const { return row_offsets_; }
    const Tensor &col_indices() const { return col_indices_; }
    const Tensor &values() const { return values_; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t nnz() const { return values_->numel(); }

    Device device() const { return values_->device(); }
    DataType dtype() const { return values_->dtype(); }
    Format format() const { return Format::CSR; }

private:
    Tensor row_offsets_; // [rows+1], I32
    Tensor col_indices_; // [nnz],   I32
    Tensor values_;      // [nnz],   T
    size_t rows_, cols_;
};

} // namespace infinicore
