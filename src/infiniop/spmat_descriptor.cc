#include "../utils.h"
#include "../utils/check.h"
#include "spmat.h"

InfiniopSpMatDescriptor::InfiniopSpMatDescriptor(
    infiniopSpMatFormat_t format,
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t ell_width,
    size_t slice_height,
    size_t sigma,
    size_t num_slices,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t crow_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    infiniopTensorDescriptor_t slice_offsets_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    void const *values,
    void const *crow_indices,
    void const *col_indices,
    void const *slice_offsets,
    void const *row_indices)
    : _format(format),
      _rows(rows),
      _cols(cols),
      _nnz(nnz),
      _ell_width(ell_width),
      _slice_height(slice_height),
      _sigma(sigma),
      _num_slices(num_slices),
      _values_desc(values_desc),
      _crow_indices_desc(crow_indices_desc),
      _col_indices_desc(col_indices_desc),
      _slice_offsets_desc(slice_offsets_desc),
      _row_indices_desc(row_indices_desc),
      _values(values),
      _crow_indices(crow_indices),
      _col_indices(col_indices),
      _slice_offsets(slice_offsets),
      _row_indices(row_indices) {}

infiniopSpMatFormat_t InfiniopSpMatDescriptor::format() const {
    return _format;
}

size_t InfiniopSpMatDescriptor::rows() const {
    return _rows;
}

size_t InfiniopSpMatDescriptor::cols() const {
    return _cols;
}

size_t InfiniopSpMatDescriptor::nnz() const {
    return _nnz;
}

size_t InfiniopSpMatDescriptor::ellWidth() const {
    return _ell_width;
}

size_t InfiniopSpMatDescriptor::sliceHeight() const {
    return _slice_height;
}

size_t InfiniopSpMatDescriptor::sigma() const {
    return _sigma;
}

size_t InfiniopSpMatDescriptor::numSlices() const {
    return _num_slices;
}

infiniopTensorDescriptor_t InfiniopSpMatDescriptor::valuesDesc() const {
    return _values_desc;
}

infiniopTensorDescriptor_t InfiniopSpMatDescriptor::crowIndicesDesc() const {
    return _crow_indices_desc;
}

infiniopTensorDescriptor_t InfiniopSpMatDescriptor::colIndicesDesc() const {
    return _col_indices_desc;
}

infiniopTensorDescriptor_t InfiniopSpMatDescriptor::sliceOffsetsDesc() const {
    return _slice_offsets_desc;
}

infiniopTensorDescriptor_t InfiniopSpMatDescriptor::rowIndicesDesc() const {
    return _row_indices_desc;
}

infiniDtype_t InfiniopSpMatDescriptor::indexDtype() const {
    if (_format == INFINIOP_SPMAT_FORMAT_CSR) {
        return _crow_indices_desc->dtype();
    }
    return _col_indices_desc->dtype();
}

void const *InfiniopSpMatDescriptor::values() const {
    return _values;
}

void const *InfiniopSpMatDescriptor::crowIndices() const {
    return _crow_indices;
}

void const *InfiniopSpMatDescriptor::colIndices() const {
    return _col_indices;
}

void const *InfiniopSpMatDescriptor::sliceOffsets() const {
    return _slice_offsets;
}

void const *InfiniopSpMatDescriptor::rowIndices() const {
    return _row_indices;
}

static infiniStatus_t checkIndexDtype(infiniopTensorDescriptor_t desc) {
    auto dtype = desc->dtype();
    CHECK_OR_RETURN(dtype == INFINI_DTYPE_I32 || dtype == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);
    return INFINI_STATUS_SUCCESS;
}

static infiniStatus_t checkPackedSparseTensors(
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    size_t packed_nnz) {
    CHECK_OR_RETURN(values_desc != nullptr && col_indices_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values_desc->numel() == packed_nnz, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(col_indices_desc->numel() == packed_nnz, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(values_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(col_indices_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_STATUS(checkIndexDtype(col_indices_desc));
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C __export infiniStatus_t infiniopCreateCsrSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t crow_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *crow_indices,
    void const *col_indices) {

    CHECK_OR_RETURN(desc_ptr != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values_desc != nullptr && crow_indices_desc != nullptr && col_indices_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values != nullptr && crow_indices != nullptr && col_indices != nullptr, INFINI_STATUS_NULL_POINTER);

    CHECK_OR_RETURN(values_desc->ndim() == 1 && values_desc->dim(0) == nnz, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(crow_indices_desc->ndim() == 1 && crow_indices_desc->dim(0) == rows + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(col_indices_desc->ndim() == 1 && col_indices_desc->dim(0) == nnz, INFINI_STATUS_BAD_TENSOR_SHAPE);

    CHECK_OR_RETURN(values_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(crow_indices_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(col_indices_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto index_dtype = crow_indices_desc->dtype();
    CHECK_OR_RETURN(index_dtype == INFINI_DTYPE_I32 || index_dtype == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(col_indices_desc->dtype() == index_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    *desc_ptr = new InfiniopSpMatDescriptor(
        INFINIOP_SPMAT_FORMAT_CSR,
        rows,
        cols,
        nnz,
        0,
        0,
        0,
        0,
        values_desc,
        crow_indices_desc,
        col_indices_desc,
        nullptr,
        nullptr,
        values,
        crow_indices,
        col_indices,
        nullptr,
        nullptr);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C __export infiniStatus_t infiniopCreateCooSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *row_indices,
    void const *col_indices) {

    CHECK_OR_RETURN(desc_ptr != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values != nullptr && row_indices != nullptr && col_indices != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_STATUS(checkPackedSparseTensors(values_desc, col_indices_desc, nnz));
    CHECK_OR_RETURN(row_indices_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(row_indices_desc->ndim() == 1 && row_indices_desc->dim(0) == nnz, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(row_indices_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(row_indices_desc->dtype() == col_indices_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);

    *desc_ptr = new InfiniopSpMatDescriptor(
        INFINIOP_SPMAT_FORMAT_COO,
        rows,
        cols,
        nnz,
        0,
        0,
        0,
        0,
        values_desc,
        nullptr,
        col_indices_desc,
        nullptr,
        row_indices_desc,
        values,
        nullptr,
        col_indices,
        nullptr,
        row_indices);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C __export infiniStatus_t infiniopCreateEllSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t ell_width,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *col_indices) {

    CHECK_OR_RETURN(desc_ptr != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values != nullptr && col_indices != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_STATUS(checkPackedSparseTensors(values_desc, col_indices_desc, rows * ell_width));
    CHECK_OR_RETURN(nnz <= rows * ell_width, INFINI_STATUS_BAD_TENSOR_SHAPE);

    *desc_ptr = new InfiniopSpMatDescriptor(
        INFINIOP_SPMAT_FORMAT_ELL,
        rows,
        cols,
        nnz,
        ell_width,
        0,
        0,
        0,
        values_desc,
        nullptr,
        col_indices_desc,
        nullptr,
        nullptr,
        values,
        nullptr,
        col_indices,
        nullptr,
        nullptr);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C __export infiniStatus_t infiniopCreateSellSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t slice_height,
    size_t num_slices,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t slice_offsets_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *slice_offsets,
    void const *col_indices) {

    CHECK_OR_RETURN(desc_ptr != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values != nullptr && slice_offsets != nullptr && col_indices != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(slice_height > 0, INFINI_STATUS_BAD_PARAM);
    CHECK_OR_RETURN(num_slices == (rows + slice_height - 1) / slice_height, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(values_desc != nullptr && col_indices_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_STATUS(checkPackedSparseTensors(values_desc, col_indices_desc, values_desc->numel()));
    CHECK_OR_RETURN(slice_offsets_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(slice_offsets_desc->ndim() == 1 && slice_offsets_desc->dim(0) == num_slices + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(slice_offsets_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_STATUS(checkIndexDtype(slice_offsets_desc));
    CHECK_OR_RETURN(col_indices_desc->dtype() == slice_offsets_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(nnz <= values_desc->numel(), INFINI_STATUS_BAD_TENSOR_SHAPE);

    *desc_ptr = new InfiniopSpMatDescriptor(
        INFINIOP_SPMAT_FORMAT_SELL,
        rows,
        cols,
        nnz,
        0,
        slice_height,
        0,
        num_slices,
        values_desc,
        nullptr,
        col_indices_desc,
        slice_offsets_desc,
        nullptr,
        values,
        nullptr,
        col_indices,
        slice_offsets,
        nullptr);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C __export infiniStatus_t infiniopCreateSellSigmaCSpMatDescriptor(
    infiniopSpMatDescriptor_t *desc_ptr,
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t slice_height,
    size_t sigma,
    size_t num_slices,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t slice_offsets_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    infiniopTensorDescriptor_t row_indices_desc,
    void const *values,
    void const *slice_offsets,
    void const *col_indices,
    void const *row_indices) {

    CHECK_OR_RETURN(desc_ptr != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values != nullptr && slice_offsets != nullptr && col_indices != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(slice_height > 0, INFINI_STATUS_BAD_PARAM);
    CHECK_OR_RETURN(num_slices == (rows + slice_height - 1) / slice_height, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(values_desc != nullptr && col_indices_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_STATUS(checkPackedSparseTensors(values_desc, col_indices_desc, values_desc->numel()));
    CHECK_OR_RETURN(slice_offsets_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(slice_offsets_desc->ndim() == 1 && slice_offsets_desc->dim(0) == num_slices + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(slice_offsets_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_STATUS(checkIndexDtype(slice_offsets_desc));
    CHECK_OR_RETURN(col_indices_desc->dtype() == slice_offsets_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(nnz <= values_desc->numel(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(row_indices_desc != nullptr && row_indices != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(row_indices_desc->ndim() == 1 && row_indices_desc->dim(0) == rows, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(row_indices_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(row_indices_desc->dtype() == col_indices_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);

    *desc_ptr = new InfiniopSpMatDescriptor(
        INFINIOP_SPMAT_FORMAT_SELL_SIGMA_C,
        rows,
        cols,
        nnz,
        0,
        slice_height,
        sigma,
        num_slices,
        values_desc,
        nullptr,
        col_indices_desc,
        slice_offsets_desc,
        row_indices_desc,
        values,
        nullptr,
        col_indices,
        slice_offsets,
        row_indices);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C __export infiniStatus_t infiniopDestroySpMatDescriptor(infiniopSpMatDescriptor_t desc) {
    delete desc;
    return INFINI_STATUS_SUCCESS;
}
