#ifndef __SPMM_H__
#define __SPMM_H__

#include "../../operator.h"
#include "info.h"

#define SPMM_DESCRIPTOR(NAMESPACE)                               \
    namespace op::spmm::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        infiniDtype_t _index_dtype;                              \
        SpMMInfo _info;                                          \
        infiniopSpMatDescriptor_t _a_desc;                       \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            infiniDtype_t index_dtype,                           \
            SpMMInfo info,                                       \
            infiniopSpMatDescriptor_t a_desc,                    \
            size_t workspace_size,                               \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _index_dtype(index_dtype),                         \
              _info(info),                                       \
              _a_desc(a_desc),                                   \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t c_desc,                   \
            infiniopSpMatDescriptor_t a_desc,                    \
            infiniopTensorDescriptor_t b_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *c,                                             \
            const void *b,                                       \
            float alpha,                                         \
            float beta,                                          \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __SPMM_H__
