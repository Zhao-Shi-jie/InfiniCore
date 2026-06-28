#ifndef __SPVV_H__
#define __SPVV_H__

#include "info.h"

#define SPVV_DESCRIPTOR(NAMESPACE)                               \
    namespace op::spvv::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        infiniDtype_t _dtype;                                    \
        infiniDtype_t _index_dtype;                              \
        SpVVInfo _info;                                          \
        infiniopSpVecDescriptor_t _a_desc;                       \
        size_t _workspace_size;                                  \
                                                                 \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            infiniDtype_t index_dtype,                           \
            SpVVInfo info,                                       \
            infiniopSpVecDescriptor_t a_desc,                    \
            size_t workspace_size,                               \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _dtype(dtype),                                     \
              _index_dtype(index_dtype),                         \
              _info(std::move(info)),                            \
              _a_desc(a_desc),                                   \
              _workspace_size(workspace_size),                   \
              _opaque(opaque) {}                                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
        size_t workspaceSize() const { return _workspace_size; } \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t y_desc,                   \
            infiniopSpVecDescriptor_t a_desc,                    \
            infiniopTensorDescriptor_t x_desc,                   \
            const void *x);                                      \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *y,                                             \
            const void *x,                                       \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __SPVV_H__
