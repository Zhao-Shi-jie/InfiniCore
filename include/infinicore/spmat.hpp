#pragma once

#include "tensor.hpp"

namespace infinicore {

class SpMatImpl;

class SpMat {
public:
    static SpMat csr(Tensor crow_indices, Tensor col_indices, Tensor values, Size rows, Size cols);

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
    ~SpMatImpl();

    Size rows() const;
    Size cols() const;
    Size nnz() const;
    DataType dtype() const;
    DataType index_dtype() const;
    Device device() const;

    const Tensor &crow_indices() const;
    const Tensor &col_indices() const;
    const Tensor &values() const;
    infiniopSpMatDescriptor_t desc() const;

private:
    Tensor crow_indices_;
    Tensor col_indices_;
    Tensor values_;
    Size rows_;
    Size cols_;
    infiniopSpMatDescriptor_t desc_;
};

} // namespace infinicore
