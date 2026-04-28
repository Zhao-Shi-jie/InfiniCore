#ifndef __INFINIOP_SPMAT_H__
#define __INFINIOP_SPMAT_H__

#include "infiniop/spmat_descriptor.h"
#include "tensor.h"

struct InfiniopSpMatDescriptor {
private:
    infiniopSpMatFormat_t _format;
    size_t _rows;
    size_t _cols;
    size_t _nnz;
    infiniopTensorDescriptor_t _values_desc;
    infiniopTensorDescriptor_t _crow_indices_desc;
    infiniopTensorDescriptor_t _col_indices_desc;
    void const *_values;
    void const *_crow_indices;
    void const *_col_indices;

public:
    InfiniopSpMatDescriptor(
        infiniopSpMatFormat_t format,
        size_t rows,
        size_t cols,
        size_t nnz,
        infiniopTensorDescriptor_t values_desc,
        infiniopTensorDescriptor_t crow_indices_desc,
        infiniopTensorDescriptor_t col_indices_desc,
        void const *values,
        void const *crow_indices,
        void const *col_indices);

    infiniopSpMatFormat_t format() const;
    size_t rows() const;
    size_t cols() const;
    size_t nnz() const;
    infiniopTensorDescriptor_t valuesDesc() const;
    infiniopTensorDescriptor_t crowIndicesDesc() const;
    infiniopTensorDescriptor_t colIndicesDesc() const;
    void const *values() const;
    void const *crowIndices() const;
    void const *colIndices() const;
};

#endif // __INFINIOP_SPMAT_H__
