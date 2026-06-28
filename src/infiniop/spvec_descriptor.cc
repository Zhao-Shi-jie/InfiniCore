#include "../utils.h"
#include "../utils/check.h"
#include "spvec.h"

InfiniopSpVecDescriptor::InfiniopSpVecDescriptor(
    size_t size,
    size_t nnz,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t indices_desc,
    void const *values,
    void const *indices)
    : _size(size),
      _nnz(nnz),
      _values_desc(values_desc),
      _indices_desc(indices_desc),
      _values(values),
      _indices(indices) {}

size_t InfiniopSpVecDescriptor::size() const {
    return _size;
}

size_t InfiniopSpVecDescriptor::nnz() const {
    return _nnz;
}

infiniopTensorDescriptor_t InfiniopSpVecDescriptor::valuesDesc() const {
    return _values_desc;
}

infiniopTensorDescriptor_t InfiniopSpVecDescriptor::indicesDesc() const {
    return _indices_desc;
}

void const *InfiniopSpVecDescriptor::values() const {
    return _values;
}

void const *InfiniopSpVecDescriptor::indices() const {
    return _indices;
}

__INFINI_C __export infiniStatus_t infiniopCreateSpVecDescriptor(
    infiniopSpVecDescriptor_t *desc_ptr,
    size_t size,
    size_t nnz,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t indices_desc,
    void const *values,
    void const *indices) {

    CHECK_OR_RETURN(desc_ptr != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values_desc != nullptr && indices_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(values != nullptr && indices != nullptr, INFINI_STATUS_NULL_POINTER);

    CHECK_OR_RETURN(values_desc->ndim() == 1 && values_desc->dim(0) == nnz, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(indices_desc->ndim() == 1 && indices_desc->dim(0) == nnz, INFINI_STATUS_BAD_TENSOR_SHAPE);

    CHECK_OR_RETURN(values_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(indices_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto index_dtype = indices_desc->dtype();
    CHECK_OR_RETURN(index_dtype == INFINI_DTYPE_I32 || index_dtype == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);

    *desc_ptr = new InfiniopSpVecDescriptor(size, nnz, values_desc, indices_desc, values, indices);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C __export infiniStatus_t infiniopDestroySpVecDescriptor(infiniopSpVecDescriptor_t desc) {
    delete desc;
    return INFINI_STATUS_SUCCESS;
}
