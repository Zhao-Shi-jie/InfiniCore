#ifndef __INFINIOP_SPMM_H__
#define __INFINIOP_SPMM_H__

#include "../../operator.h"
#include "info.h"

/**
 * DESCRIPTOR(NAMESPACE) defines a Descriptor class for SpMM in the given
 * hardware namespace, following the same PImpl pattern used by gemm.
 *
 * Each backend namespace provides its own `create` and `calculate`
 * implementations while sharing the common descriptor fields.
 */

#define DESCRIPTOR(NAMESPACE)                                       \
                                                                    \
    namespace op::spmm::NAMESPACE {                                 \
    class Descriptor final : public InfiniopDescriptor {            \
        struct Opaque;                                              \
        Opaque *_opaque;                                            \
        infiniDtype_t _dtype;                                       \
        SpmmInfo _info;                                             \
        size_t _workspace_size;                                     \
                                                                    \
        Descriptor(                                                 \
            infiniDtype_t dtype,                                    \
            SpmmInfo info,                                          \
            size_t workspace_size_,                                 \
            Opaque *opaque,                                         \
            infiniDevice_t device_type,                             \
            int device_id)                                          \
            : InfiniopDescriptor{device_type, device_id},           \
              _opaque(opaque),                                      \
              _dtype(dtype),                                        \
              _info(info),                                          \
              _workspace_size(workspace_size_) {}                   \
                                                                    \
    public:                                                         \
        ~Descriptor();                                              \
                                                                    \
        size_t workspaceSize() const { return _workspace_size; }    \
                                                                    \
        static infiniStatus_t create(                               \
            infiniopHandle_t handle,                                \
            Descriptor **desc_ptr,                                  \
            infiniopTensorDescriptor_t c_desc,                      \
            infiniopTensorDescriptor_t b_desc,                      \
            infiniopTensorDescriptor_t values_desc,                 \
            size_t rows,                                            \
            size_t cols);                                           \
                                                                    \
        infiniStatus_t calculate(                                   \
            void *workspace, size_t workspace_size,                 \
            void *c,                                                \
            const void *row_offsets,                                \
            const void *col_indices,                                \
            const void *values,                                     \
            const void *b,                                          \
            float alpha, float beta,                                \
            void *stream) const;                                    \
    };                                                              \
    }

#endif // __INFINIOP_SPMM_H__
