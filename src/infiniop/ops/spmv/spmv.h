#ifndef __SPMV_H__
#define __SPMV_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::spmv::NAMESPACE {                       \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
        SpMVInfo _info;                                   \
                                                          \
        Descriptor(                                       \
            SpMVInfo info,                                \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _info(info) {}                              \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t y_desc,            \
            size_t num_cols,                              \
            size_t num_rows,                              \
            size_t nnz);                                  \
                                                          \
        infiniStatus_t calculate(                         \
            void *y,                                      \
            const void *x,                                \
            const void *values,                           \
            const void *row_ptr,                          \
            const void *col_indices,                      \
            void *stream) const;                          \
    };                                                    \
    }


// SpMV operation information
class SpMVInfo {
    SpMVInfo() = default;

public:
    infiniDtype_t dtype;
    size_t num_rows;
    size_t num_cols;
    size_t nnz;

    static utils::Result<SpMVInfo> create(
        infiniopTensorDescriptor_t y_desc,
        size_t num_cols,
        size_t num_rows,
        size_t nnz) {

        CHECK_OR_RETURN(num_cols > 0 && num_rows > 0 && nnz > 0,
                        INFINI_STATUS_BAD_PARAM);

        SpMVInfo info;
        info.num_rows = num_rows;
        info.num_cols = num_cols;
        info.nnz = nnz;
        info.dtype = y_desc->dtype();

        return utils::Result<SpMVInfo>(info);
    }

    static utils::Result<SpMVInfo> createLegacy(
        infiniopTensorDescriptor_t y_desc,
        size_t num_rows,
        size_t num_cols,
        size_t nnz) {

        CHECK_OR_RETURN(num_rows > 0 && num_cols > 0 && nnz > 0,
                        INFINI_STATUS_BAD_PARAM);

        SpMVInfo info;
        info.num_rows = num_rows;
        info.num_cols = num_cols;
        info.nnz = nnz;
        info.dtype = y_desc->dtype();

        return utils::Result<SpMVInfo>(info);
    }
};

// validate SpMV CSR operation parameters
inline infiniStatus_t validateSpMVCSR(
    const void *y, const void *x, const void *values,
    const void *row_indices, const void *col_indices) {

    CHECK_OR_RETURN(y && x && values && row_indices && col_indices,
                    INFINI_STATUS_BAD_PARAM);
    // CHECK_OR_RETURN(dtype == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);

    return INFINI_STATUS_SUCCESS;
}

#endif
