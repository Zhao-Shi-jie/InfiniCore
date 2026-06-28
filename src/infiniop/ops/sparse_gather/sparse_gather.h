#ifndef __SPARSE_GATHER_H__
#define __SPARSE_GATHER_H__

#include "../../operator.h"
#include "info.h"

#define SPARSE_GATHER_DESCRIPTOR(NAMESPACE)                      \
    namespace op::sparse_gather::NAMESPACE {                     \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        infiniDtype_t _index_dtype;                              \
        SparseGatherInfo _info;                                  \
        infiniopSpVecDescriptor_t _pattern_desc;                 \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            infiniDtype_t index_dtype,                           \
            SparseGatherInfo info,                               \
            infiniopSpVecDescriptor_t pattern_desc,              \
            size_t workspace_size,                               \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _index_dtype(index_dtype),                         \
              _info(std::move(info)),                            \
              _pattern_desc(pattern_desc),                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
        size_t workspaceSize() const { return _workspace_size; } \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t output_desc,              \
            infiniopSpVecDescriptor_t pattern_desc,              \
            infiniopTensorDescriptor_t input_desc,               \
            void *output,                                        \
            const void *input);                                  \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *input,                                   \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __SPARSE_GATHER_H__
