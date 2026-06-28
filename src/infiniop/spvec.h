#ifndef __INFINIOP_SPVEC_H__
#define __INFINIOP_SPVEC_H__

#include "infiniop/spvec_descriptor.h"
#include "tensor.h"

struct InfiniopSpVecDescriptor {
private:
    size_t _size;
    size_t _nnz;
    infiniopTensorDescriptor_t _values_desc;
    infiniopTensorDescriptor_t _indices_desc;
    void const *_values;
    void const *_indices;

public:
    InfiniopSpVecDescriptor(
        size_t size,
        size_t nnz,
        infiniopTensorDescriptor_t values_desc,
        infiniopTensorDescriptor_t indices_desc,
        void const *values,
        void const *indices);

    size_t size() const;
    size_t nnz() const;
    infiniopTensorDescriptor_t valuesDesc() const;
    infiniopTensorDescriptor_t indicesDesc() const;
    void const *values() const;
    void const *indices() const;
};

#endif // __INFINIOP_SPVEC_H__
