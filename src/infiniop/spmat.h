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
    size_t _ell_width;
    size_t _slice_height;
    size_t _sigma;
    size_t _num_slices;
    infiniopTensorDescriptor_t _values_desc;
    infiniopTensorDescriptor_t _crow_indices_desc;
    infiniopTensorDescriptor_t _col_indices_desc;
    infiniopTensorDescriptor_t _slice_offsets_desc;
    infiniopTensorDescriptor_t _row_indices_desc;
    void const *_values;
    void const *_crow_indices;
    void const *_col_indices;
    void const *_slice_offsets;
    void const *_row_indices;

public:
    InfiniopSpMatDescriptor(
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
        void const *row_indices);

    infiniopSpMatFormat_t format() const;
    size_t rows() const;
    size_t cols() const;
    size_t nnz() const;
    size_t ellWidth() const;
    size_t sliceHeight() const;
    size_t sigma() const;
    size_t numSlices() const;
    infiniopTensorDescriptor_t valuesDesc() const;
    infiniopTensorDescriptor_t crowIndicesDesc() const;
    infiniopTensorDescriptor_t colIndicesDesc() const;
    infiniopTensorDescriptor_t sliceOffsetsDesc() const;
    infiniopTensorDescriptor_t rowIndicesDesc() const;
    infiniDtype_t indexDtype() const;
    void const *values() const;
    void const *crowIndices() const;
    void const *colIndices() const;
    void const *sliceOffsets() const;
    void const *rowIndices() const;
};

#endif // __INFINIOP_SPMAT_H__
