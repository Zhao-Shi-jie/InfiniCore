#include "../../../utils.h"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/hardsigmoid.hpp"
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <vector>

namespace infinicore::op::hardsigmoid_impl::cpu {

// Hardsigmoid 公式: hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
constexpr float ALPHA = 1.0f / 6.0f;
constexpr float BETA = 0.5f; // 3/6 = 0.5

template <typename T>
inline T hardsigmoid_scalar(T x) {
    float val = utils::cast<float>(x);
    float result = ALPHA * val + BETA;
    result = std::max(0.0f, std::min(1.0f, result));
    return utils::cast<T>(result);
}

// Contiguous tensor 的快速实现
template <typename T>
void hardsigmoid_contiguous(T *output_ptr, const T *input_ptr, size_t numel) {
#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        output_ptr[i] = hardsigmoid_scalar(input_ptr[i]);
    }
}

// Non-contiguous tensor 的实现（支持任意 stride）
template <typename T>
void hardsigmoid_strided(T *output_base, const T *input_base,
                         const std::vector<size_t> &shape,
                         const std::vector<int64_t> &input_strides,
                         const std::vector<int64_t> &output_strides,
                         size_t numel, int ndim) {
#pragma omp parallel for
    for (size_t linear_idx = 0; linear_idx < numel; ++linear_idx) {
        // 将线性索引转换为多维坐标
        size_t temp_idx = linear_idx;
        size_t input_offset = 0;
        size_t output_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp_idx % shape[d];
            temp_idx /= shape[d];
            input_offset += coord * input_strides[d];
            output_offset += coord * output_strides[d];
        }

        output_base[output_offset] = hardsigmoid_scalar(input_base[input_offset]);
    }
}

// 检查 tensor 是否是 contiguous
bool is_contiguous(const std::vector<size_t> &shape,
                   const std::vector<int64_t> &strides) {
    int ndim = shape.size();
    int64_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (strides[i] != expected_stride && shape[i] > 1) {
            return false;
        }
        expected_stride *= shape[i];
    }
    return true;
}

void calculate(Tensor output, Tensor input) {
    auto dtype = input->dtype();
    size_t numel = input->numel();
    auto input_base = input->data();
    auto output_base = output->data();

    auto shape = input->shape();
    auto input_strides = input->strides();
    auto output_strides = output->strides();
    int ndim = input->ndim();

    bool input_contiguous = is_contiguous(shape, input_strides);
    bool output_contiguous = is_contiguous(shape, output_strides);
    bool both_contiguous = input_contiguous && output_contiguous;

    // F32 实现
    if (dtype == DataType::F32) {
        auto *in_ptr = reinterpret_cast<const float *>(input_base);
        auto *out_ptr = reinterpret_cast<float *>(output_base);

        if (both_contiguous) {
            hardsigmoid_contiguous<float>(out_ptr, in_ptr, numel);
        } else {
            hardsigmoid_strided<float>(out_ptr, in_ptr, shape,
                                       input_strides, output_strides,
                                       numel, ndim);
        }
    }
    // F16 实现
    else if (dtype == DataType::F16) {
        auto *in_ptr = reinterpret_cast<const fp16_t *>(input_base);
        auto *out_ptr = reinterpret_cast<fp16_t *>(output_base);

        if (both_contiguous) {
            hardsigmoid_contiguous<fp16_t>(out_ptr, in_ptr, numel);
        } else {
            hardsigmoid_strided<fp16_t>(out_ptr, in_ptr, shape,
                                        input_strides, output_strides,
                                        numel, ndim);
        }
    }
    // BF16 实现
    else if (dtype == DataType::BF16) {
        auto *in_ptr = reinterpret_cast<const bf16_t *>(input_base);
        auto *out_ptr = reinterpret_cast<bf16_t *>(output_base);

        if (both_contiguous) {
            hardsigmoid_contiguous<bf16_t>(out_ptr, in_ptr, numel);
        } else {
            hardsigmoid_strided<bf16_t>(out_ptr, in_ptr, shape,
                                        input_strides, output_strides,
                                        numel, ndim);
        }
    }
    // F64 实现
    else if (dtype == DataType::F64) {
        auto *in_ptr = reinterpret_cast<const double *>(input_base);
        auto *out_ptr = reinterpret_cast<double *>(output_base);

        if (both_contiguous) {
            hardsigmoid_contiguous<double>(out_ptr, in_ptr, numel);
        } else {
            hardsigmoid_strided<double>(out_ptr, in_ptr, shape,
                                        input_strides, output_strides,
                                        numel, ndim);
        }
    } else {
        throw std::runtime_error("Unsupported dtype for hardsigmoid.");
    }
}

// 自动注册到调度器
static bool registered = []() {
    Hardsigmoid::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::hardsigmoid_impl::cpu
