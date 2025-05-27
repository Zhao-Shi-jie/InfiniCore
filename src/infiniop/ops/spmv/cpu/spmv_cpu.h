#ifndef __SPMV_CPU_H__
#define __SPMV_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../info.h"
#include <memory>

namespace op::spmv::cpu {

class Descriptor final : public operator_base {
private:
    SpMVInfo _info;

public:
    Descriptor(const SpMVInfo &info, infiniDevice_t device, int32_t device_id)
        : operator_base{device, device_id}, _info(info) {}
    ~Descriptor() = default;

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t values_desc,
        infiniopTensorDescriptor_t row_indices_desc,
        infiniopTensorDescriptor_t col_indices_desc,
        SparseFormat format);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        const void *values,
        const void *row_indices,
        const void *col_indices,
        void *stream) const;
};

} // namespace op::spmv::cpu

#endif // __SPMV_CPU_H__
