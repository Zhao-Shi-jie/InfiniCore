#include "infinicore/spmat.hpp"
#include "utils.hpp"

namespace infinicore {

SpMat SpMat::csr(Tensor crow_indices, Tensor col_indices, Tensor values, Size rows, Size cols) {
    return SpMat{std::make_shared<SpMatImpl>(crow_indices, col_indices, values, rows, cols)};
}

SpMatImpl *SpMat::operator->() {
    return impl_.get();
}

const SpMatImpl *SpMat::operator->() const {
    return impl_.get();
}

SpMat::operator bool() const {
    return impl_ != nullptr;
}

SpMatImpl::SpMatImpl(Tensor crow_indices, Tensor col_indices, Tensor values, Size rows, Size cols)
    : crow_indices_(crow_indices),
      col_indices_(col_indices),
      values_(values),
      rows_(rows),
      cols_(cols),
      desc_(nullptr) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(crow_indices_, col_indices_, values_);
    INFINICORE_CHECK_ERROR(infiniopCreateCsrSpMatDescriptor(
        &desc_,
        rows_,
        cols_,
        values_->numel(),
        values_->desc(),
        crow_indices_->desc(),
        col_indices_->desc(),
        values_->data(),
        crow_indices_->data(),
        col_indices_->data()));
}

SpMatImpl::~SpMatImpl() {
    if (desc_) {
        infiniopDestroySpMatDescriptor(desc_);
        desc_ = nullptr;
    }
}

Size SpMatImpl::rows() const {
    return rows_;
}

Size SpMatImpl::cols() const {
    return cols_;
}

Size SpMatImpl::nnz() const {
    return values_->numel();
}

DataType SpMatImpl::dtype() const {
    return values_->dtype();
}

DataType SpMatImpl::index_dtype() const {
    return crow_indices_->dtype();
}

Device SpMatImpl::device() const {
    return values_->device();
}

const Tensor &SpMatImpl::crow_indices() const {
    return crow_indices_;
}

const Tensor &SpMatImpl::col_indices() const {
    return col_indices_;
}

const Tensor &SpMatImpl::values() const {
    return values_;
}

infiniopSpMatDescriptor_t SpMatImpl::desc() const {
    return desc_;
}

} // namespace infinicore
