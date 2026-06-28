#pragma once

#include "tensor.hpp"

namespace infinicore {

class SpVecImpl;

class SpVec {
public:
    static SpVec coo(Tensor indices, Tensor values, Size size);

    SpVec() = default;
    SpVec(const SpVec &) = default;
    SpVec(SpVec &&) = default;
    SpVec &operator=(const SpVec &) = default;
    SpVec &operator=(SpVec &&) = default;

    SpVecImpl *operator->();
    const SpVecImpl *operator->() const;

    operator bool() const;

private:
    explicit SpVec(std::shared_ptr<SpVecImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<SpVecImpl> impl_;
};

class SpVecImpl {
public:
    SpVecImpl(Tensor indices, Tensor values, Size size);
    ~SpVecImpl();

    Size size() const;
    Size nnz() const;
    DataType dtype() const;
    DataType index_dtype() const;
    Device device() const;

    const Tensor &indices() const;
    const Tensor &values() const;
    infiniopSpVecDescriptor_t desc() const;

private:
    Tensor indices_;
    Tensor values_;
    Size size_;
    infiniopSpVecDescriptor_t desc_;
};

} // namespace infinicore
