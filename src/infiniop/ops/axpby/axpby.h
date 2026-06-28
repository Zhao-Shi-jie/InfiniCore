#ifndef __AXPBY_H__
#define __AXPBY_H__

#include "../../operator.h"
#include "info.h"

#define AXPBY_DESCRIPTOR(NAMESPACE)                              \
    namespace op::axpby::NAMESPACE {                             \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AxpbyInfo _info;                                         \
        infiniopSpVecDescriptor_t _x_desc;                       \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            AxpbyInfo info,                                      \
            infiniopSpVecDescriptor_t x_desc,                    \
            size_t workspace_size,                               \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(std::move(info)),                            \
              _x_desc(x_desc),                                   \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
        size_t workspaceSize() const { return _workspace_size; } \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopSpVecDescriptor_t x_desc,                    \
            infiniopTensorDescriptor_t y_desc);                  \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *y,                                             \
            float alpha,                                         \
            float beta,                                          \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __AXPBY_H__
