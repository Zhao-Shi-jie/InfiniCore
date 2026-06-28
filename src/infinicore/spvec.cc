#include "infinicore/spvec.hpp"
#include "utils.hpp"

namespace infinicore {

SpVec SpVec::coo(Tensor indices, Tensor values, Size size) {
    return SpVec{std::make_shared<SpVecImpl>(indices, values, size)};
}

SpVecImpl *SpVec::operator->() {
    return impl_.get();
}

const SpVecImpl *SpVec::operator->() const {
    return impl_.get();
}

SpVec::operator bool() const {
    return impl_ != nullptr;
}

SpVecImpl::SpVecImpl(Tensor indices, Tensor values, Size size)
    : indices_(indices),
      values_(values),
      size_(size),
      desc_(nullptr) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices_, values_);
    INFINICORE_ASSERT(indices_->is_contiguous());
    INFINICORE_ASSERT(values_->is_contiguous());
    INFINICORE_CHECK_ERROR(infiniopCreateSpVecDescriptor(
        &desc_,
        size_,
        values_->numel(),
        values_->desc(),
        indices_->desc(),
        values_->data(),
        indices_->data()));
}

SpVecImpl::~SpVecImpl() {
    if (desc_) {
        infiniopDestroySpVecDescriptor(desc_);
        desc_ = nullptr;
    }
}

Size SpVecImpl::size() const {
    return size_;
}

Size SpVecImpl::nnz() const {
    return values_->numel();
}

DataType SpVecImpl::dtype() const {
    return values_->dtype();
}

DataType SpVecImpl::index_dtype() const {
    return indices_->dtype();
}

Device SpVecImpl::device() const {
    return values_->device();
}

const Tensor &SpVecImpl::indices() const {
    return indices_;
}

const Tensor &SpVecImpl::values() const {
    return values_;
}

infiniopSpVecDescriptor_t SpVecImpl::desc() const {
    return desc_;
}

} // namespace infinicore
