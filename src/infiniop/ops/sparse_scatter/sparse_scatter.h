#ifndef __SPARSE_SCATTER_H__
#define __SPARSE_SCATTER_H__

#include "../../operator.h"
#include "info.h"

#define SPARSE_SCATTER_DESCRIPTOR(NAMESPACE)                     \
    namespace op::sparse_scatter::NAMESPACE {                    \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        infiniDtype_t _index_dtype;                              \
        SparseScatterInfo _info;                                 \
        infiniopSpVecDescriptor_t _input_desc;                   \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            infiniDtype_t index_dtype,                           \
            SparseScatterInfo info,                              \
            infiniopSpVecDescriptor_t input_desc,                \
            size_t workspace_size,                               \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _index_dtype(index_dtype),                         \
              _info(std::move(info)),                            \
              _input_desc(input_desc),                           \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
        size_t workspaceSize() const { return _workspace_size; } \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t output_desc,              \
            infiniopSpVecDescriptor_t input_desc);               \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __SPARSE_SCATTER_H__
