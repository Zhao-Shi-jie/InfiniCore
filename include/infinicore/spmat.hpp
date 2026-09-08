#pragma once

#include "tensor.hpp"

namespace infinicore {

class SpMatImpl;

class SpMat {
public:
    static SpMat csr(Tensor crow_indices, Tensor col_indices, Tensor values, Size rows, Size cols);
    static SpMat coo(Tensor row_indices, Tensor col_indices, Tensor values, Size rows, Size cols);
    static SpMat ell(Tensor col_indices, Tensor values, Size rows, Size cols, Size ell_width, Size nnz);
    static SpMat sell(Tensor slice_offsets, Tensor col_indices, Tensor values, Size rows, Size cols, Size slice_height, Size nnz);
    static SpMat sell_sigma_c(Tensor slice_offsets, Tensor col_indices, Tensor row_indices, Tensor values, Size rows, Size cols, Size slice_height, Size sigma, Size nnz);

    SpMat() = default;
    SpMat(const SpMat &) = default;
    SpMat(SpMat &&) = default;
    SpMat &operator=(const SpMat &) = default;
    SpMat &operator=(SpMat &&) = default;

    SpMatImpl *operator->();
    const SpMatImpl *operator->() const;

    operator bool() const;

private:
    explicit SpMat(std::shared_ptr<SpMatImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<SpMatImpl> impl_;
};

class SpMatImpl {
public:
    SpMatImpl(Tensor crow_indices, Tensor col_indices, Tensor values, Size rows, Size cols);
    SpMatImpl(Tensor row_indices, Tensor col_indices, Tensor values, Size rows, Size cols, infiniopSpMatFormat_t format);
    SpMatImpl(Tensor col_indices, Tensor values, Size rows, Size cols, Size ell_width, Size nnz);
    SpMatImpl(Tensor slice_offsets, Tensor col_indices, Tensor values, Size rows, Size cols, Size slice_height, Size nnz);
    SpMatImpl(Tensor slice_offsets, Tensor col_indices, Tensor row_indices, Tensor values, Size rows, Size cols, Size slice_height, Size sigma, Size nnz);
    ~SpMatImpl();

    infiniopSpMatFormat_t format() const;
    Size rows() const;
    Size cols() const;
    Size nnz() const;
    Size ell_width() const;
    Size slice_height() const;
    Size sigma() const;
    DataType dtype() const;
    DataType index_dtype() const;
    Device device() const;

    const Tensor &crow_indices() const;
    const Tensor &col_indices() const;
    const Tensor &slice_offsets() const;
    const Tensor &row_indices() const;
    const Tensor &values() const;
    infiniopSpMatDescriptor_t desc() const;

private:
    infiniopSpMatFormat_t format_;
    Tensor crow_indices_;
    Tensor col_indices_;
    Tensor slice_offsets_;
    Tensor row_indices_;
    Tensor values_;
    Size rows_;
    Size cols_;
    Size nnz_;
    Size ell_width_;
    Size slice_height_;
    Size sigma_;
    infiniopSpMatDescriptor_t desc_;
};

} // namespace infinicore
