#include "infinicore/spmat.hpp"
#include "utils.hpp"

namespace infinicore {

SpMat SpMat::csr(Tensor crow_indices, Tensor col_indices, Tensor values, Size rows, Size cols) {
    return SpMat{std::make_shared<SpMatImpl>(crow_indices, col_indices, values, rows, cols)};
}

SpMat SpMat::coo(Tensor row_indices, Tensor col_indices, Tensor values, Size rows, Size cols) {
    return SpMat{std::make_shared<SpMatImpl>(row_indices, col_indices, values, rows, cols, INFINIOP_SPMAT_FORMAT_COO)};
}

SpMat SpMat::ell(Tensor col_indices, Tensor values, Size rows, Size cols, Size ell_width, Size nnz) {
    return SpMat{std::make_shared<SpMatImpl>(col_indices, values, rows, cols, ell_width, nnz)};
}

SpMat SpMat::sell(Tensor slice_offsets, Tensor col_indices, Tensor values, Size rows, Size cols, Size slice_height, Size nnz) {
    return SpMat{std::make_shared<SpMatImpl>(slice_offsets, col_indices, values, rows, cols, slice_height, nnz)};
}

SpMat SpMat::sell_sigma_c(Tensor slice_offsets, Tensor col_indices, Tensor row_indices, Tensor values, Size rows, Size cols, Size slice_height, Size sigma, Size nnz) {
    return SpMat{std::make_shared<SpMatImpl>(slice_offsets, col_indices, row_indices, values, rows, cols, slice_height, sigma, nnz)};
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
    : format_(INFINIOP_SPMAT_FORMAT_CSR),
      crow_indices_(crow_indices),
      col_indices_(col_indices),
      values_(values),
      rows_(rows),
      cols_(cols),
      nnz_(values->numel()),
      ell_width_(0),
      slice_height_(0),
      sigma_(0),
      desc_(nullptr) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(crow_indices_, col_indices_, values_);
    INFINICORE_ASSERT(crow_indices_->is_contiguous());
    INFINICORE_ASSERT(col_indices_->is_contiguous());
    INFINICORE_ASSERT(values_->is_contiguous());
    INFINICORE_CHECK_ERROR(infiniopCreateCsrSpMatDescriptor(
        &desc_,
        rows_,
        cols_,
        nnz_,
        values_->desc(),
        crow_indices_->desc(),
        col_indices_->desc(),
        values_->data(),
        crow_indices_->data(),
        col_indices_->data()));
}

SpMatImpl::SpMatImpl(Tensor row_indices, Tensor col_indices, Tensor values, Size rows, Size cols, infiniopSpMatFormat_t format)
    : format_(format),
      col_indices_(col_indices),
      row_indices_(row_indices),
      values_(values),
      rows_(rows),
      cols_(cols),
      nnz_(values->numel()),
      ell_width_(0),
      slice_height_(0),
      sigma_(0),
      desc_(nullptr) {
    INFINICORE_ASSERT(format_ == INFINIOP_SPMAT_FORMAT_COO);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(row_indices_, col_indices_, values_);
    INFINICORE_ASSERT(row_indices_->is_contiguous());
    INFINICORE_ASSERT(col_indices_->is_contiguous());
    INFINICORE_ASSERT(values_->is_contiguous());
    INFINICORE_CHECK_ERROR(infiniopCreateCooSpMatDescriptor(
        &desc_,
        rows_,
        cols_,
        nnz_,
        values_->desc(),
        row_indices_->desc(),
        col_indices_->desc(),
        values_->data(),
        row_indices_->data(),
        col_indices_->data()));
}

SpMatImpl::SpMatImpl(Tensor col_indices, Tensor values, Size rows, Size cols, Size ell_width, Size nnz)
    : format_(INFINIOP_SPMAT_FORMAT_ELL),
      col_indices_(col_indices),
      values_(values),
      rows_(rows),
      cols_(cols),
      nnz_(nnz),
      ell_width_(ell_width),
      slice_height_(0),
      sigma_(0),
      desc_(nullptr) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(col_indices_, values_);
    INFINICORE_ASSERT(col_indices_->is_contiguous());
    INFINICORE_ASSERT(values_->is_contiguous());
    INFINICORE_CHECK_ERROR(infiniopCreateEllSpMatDescriptor(
        &desc_,
        rows_,
        cols_,
        nnz_,
        ell_width_,
        values_->desc(),
        col_indices_->desc(),
        values_->data(),
        col_indices_->data()));
}

SpMatImpl::SpMatImpl(Tensor slice_offsets, Tensor col_indices, Tensor values, Size rows, Size cols, Size slice_height, Size nnz)
    : format_(INFINIOP_SPMAT_FORMAT_SELL),
      col_indices_(col_indices),
      slice_offsets_(slice_offsets),
      values_(values),
      rows_(rows),
      cols_(cols),
      nnz_(nnz),
      ell_width_(0),
      slice_height_(slice_height),
      sigma_(0),
      desc_(nullptr) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(slice_offsets_, col_indices_, values_);
    INFINICORE_ASSERT(slice_offsets_->is_contiguous());
    INFINICORE_ASSERT(col_indices_->is_contiguous());
    INFINICORE_ASSERT(values_->is_contiguous());
    INFINICORE_ASSERT(slice_offsets_->numel() > 0);
    INFINICORE_CHECK_ERROR(infiniopCreateSellSpMatDescriptor(
        &desc_,
        rows_,
        cols_,
        nnz_,
        slice_height_,
        slice_offsets_->numel() - 1,
        values_->desc(),
        slice_offsets_->desc(),
        col_indices_->desc(),
        values_->data(),
        slice_offsets_->data(),
        col_indices_->data()));
}

SpMatImpl::SpMatImpl(Tensor slice_offsets, Tensor col_indices, Tensor row_indices, Tensor values, Size rows, Size cols, Size slice_height, Size sigma, Size nnz)
    : format_(INFINIOP_SPMAT_FORMAT_SELL_SIGMA_C),
      col_indices_(col_indices),
      slice_offsets_(slice_offsets),
      row_indices_(row_indices),
      values_(values),
      rows_(rows),
      cols_(cols),
      nnz_(nnz),
      ell_width_(0),
      slice_height_(slice_height),
      sigma_(sigma),
      desc_(nullptr) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(slice_offsets_, col_indices_, row_indices_, values_);
    INFINICORE_ASSERT(slice_offsets_->is_contiguous());
    INFINICORE_ASSERT(col_indices_->is_contiguous());
    INFINICORE_ASSERT(row_indices_->is_contiguous());
    INFINICORE_ASSERT(values_->is_contiguous());
    INFINICORE_ASSERT(slice_offsets_->numel() > 0);
    INFINICORE_CHECK_ERROR(infiniopCreateSellSigmaCSpMatDescriptor(
        &desc_,
        rows_,
        cols_,
        nnz_,
        slice_height_,
        sigma_,
        slice_offsets_->numel() - 1,
        values_->desc(),
        slice_offsets_->desc(),
        col_indices_->desc(),
        row_indices_->desc(),
        values_->data(),
        slice_offsets_->data(),
        col_indices_->data(),
        row_indices_->data()));
}

SpMatImpl::~SpMatImpl() {
    if (desc_) {
        infiniopDestroySpMatDescriptor(desc_);
        desc_ = nullptr;
    }
}

infiniopSpMatFormat_t SpMatImpl::format() const {
    return format_;
}

Size SpMatImpl::rows() const {
    return rows_;
}

Size SpMatImpl::cols() const {
    return cols_;
}

Size SpMatImpl::nnz() const {
    return nnz_;
}

Size SpMatImpl::ell_width() const {
    return ell_width_;
}

Size SpMatImpl::slice_height() const {
    return slice_height_;
}

Size SpMatImpl::sigma() const {
    return sigma_;
}

DataType SpMatImpl::dtype() const {
    return values_->dtype();
}

DataType SpMatImpl::index_dtype() const {
    if (format_ == INFINIOP_SPMAT_FORMAT_CSR) {
        return crow_indices_->dtype();
    }
    return col_indices_->dtype();
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

const Tensor &SpMatImpl::slice_offsets() const {
    return slice_offsets_;
}

const Tensor &SpMatImpl::row_indices() const {
    return row_indices_;
}

const Tensor &SpMatImpl::values() const {
    return values_;
}

infiniopSpMatDescriptor_t SpMatImpl::desc() const {
    return desc_;
}

} // namespace infinicore
