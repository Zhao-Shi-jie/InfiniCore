#ifndef __SDDMM_H__
#define __SDDMM_H__

#include "../../operator.h"
#include "info.h"

#define SDDMM_DESCRIPTOR(NAMESPACE)                              \
    namespace op::sddmm::NAMESPACE {                             \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        infiniDtype_t _index_dtype;                              \
        SDDMMInfo _info;                                         \
        infiniopSpMatDescriptor_t _c_desc;                       \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            infiniDtype_t index_dtype,                           \
            SDDMMInfo info,                                      \
            infiniopSpMatDescriptor_t c_desc,                    \
            size_t workspace_size,                               \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _index_dtype(index_dtype),                         \
              _info(std::move(info)),                            \
              _c_desc(c_desc),                                   \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
        size_t workspaceSize() const { return _workspace_size; } \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopSpMatDescriptor_t c_desc,                    \
            infiniopTensorDescriptor_t a_desc,                   \
            infiniopTensorDescriptor_t b_desc);                  \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *c_values,                                      \
            const void *a,                                       \
            const void *b,                                       \
            float alpha,                                         \
            float beta,                                          \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __SDDMM_H__
