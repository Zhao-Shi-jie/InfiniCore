#include "../utils.h"
#include "../utils/check.h"
#include "spmat.h"

InfiniopSpMatDescriptor::InfiniopSpMatDescriptor(
    infiniopSpMatFormat_t format,
    size_t rows,
    size_t cols,
    size_t nnz,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t crow_indices_desc,
    infiniopTensorDescriptor_t col_indices_desc,
    void const *values,
    void const *crow_indices,
    void const *col_indices)
    : _format(format),
      _rows(rows),
      _cols(cols),
      _nnz(nnz),
      _values_desc(values_desc),
      _crow_indices_desc(crow_indices_desc),
      _col_indices_desc(col_indices_desc),
      _values(values),
      _crow_indices(crow_indices),
      _col_indices(col_indices) {}

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

infiniopTensorDescriptor_t InfiniopSpMatDescriptor::valuesDesc() const {
    return _values_desc;
}

infiniopTensorDescriptor_t InfiniopSpMatDescriptor::crowIndicesDesc() const {
    return _crow_indices_desc;
}

infiniopTensorDescriptor_t InfiniopSpMatDescriptor::colIndicesDesc() const {
    return _col_indices_desc;
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
        values_desc,
        crow_indices_desc,
        col_indices_desc,
        values,
        crow_indices,
        col_indices);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C __export infiniStatus_t infiniopDestroySpMatDescriptor(infiniopSpMatDescriptor_t desc) {
    delete desc;
    return INFINI_STATUS_SUCCESS;
}
