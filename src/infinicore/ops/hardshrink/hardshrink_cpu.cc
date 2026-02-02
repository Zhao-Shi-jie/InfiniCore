#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/hardshrink.hpp"
#include <cmath>
#include <omp.h>
#include <vector>

namespace infinicore::op::hardshrink_impl::cpu {

using namespace infinicore;

template <typename T>
void hardshrink_contiguous(T *output_data,
                           const T *input_data,
                           size_t numel,
                           float lambd) {
#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        float val = utils::cast<float>(input_data[i]);
        float result;
        if (val > lambd) {
            result = val;
        } else if (val < -lambd) {
            result = val;
        } else {
            result = 0.0f;
        }
        output_data[i] = utils::cast<T>(result);
    }
}

// strided 版本
template <typename T>
void hardshrink_strided(T *output_base,
                        const T *input_base,
                        const std::vector<size_t> &shape,
                        const std::vector<int64_t> &input_strides,
                        const std::vector<int64_t> &output_strides,
                        float lambd) {
    int ndim = shape.size();
    size_t numel = 1;
    for (int i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }

#pragma omp parallel for
    for (size_t idx = 0; idx < numel; ++idx) {
        size_t temp = idx;
        size_t input_offset = 0;
        size_t output_offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp % shape[d];
            temp /= shape[d];
            input_offset += coord * input_strides[d];
            output_offset += coord * output_strides[d];
        }

        float val = utils::cast<float>(input_base[input_offset]);
        float result;
        if (val > lambd) {
            result = val;
        } else if (val < -lambd) {
            result = val;
        } else {
            result = 0.0f;
        }
        output_base[output_offset] = utils::cast<T>(result);
    }
}

template <typename T>
void calculate_typed(Tensor output, Tensor input, float lambd) {
    auto *input_ptr = reinterpret_cast<const T *>(input->data());
    auto *output_ptr = reinterpret_cast<T *>(output->data());

    size_t numel = 1;
    auto shape = input->shape();
    for (size_t s : shape) {
        numel *= s;
    }

    if (input->is_contiguous() && output->is_contiguous()) {
        hardshrink_contiguous<T>(output_ptr, input_ptr, numel, lambd);
    } else {
        hardshrink_strided<T>(output_ptr, input_ptr,
                              shape,
                              input->strides(),
                              output->strides(),
                              lambd);
    }
}

void calculate(Tensor output, Tensor input, float lambd) {
    auto dtype = input->dtype();

    if (dtype == DataType::F32) {
        calculate_typed<float>(output, input, lambd);
    } else if (dtype == DataType::F64) {
        calculate_typed<double>(output, input, lambd);
    } else if (dtype == DataType::F16) {
        calculate_typed<fp16_t>(output, input, lambd);
    } else if (dtype == DataType::BF16) {
        calculate_typed<bf16_t>(output, input, lambd);
    } else if (dtype == DataType::I32) {
        calculate_typed<int32_t>(output, input, lambd);
    } else {
        throw std::runtime_error("Unsupported dtype for hardshrink.");
    }
}

static bool registered = []() {
    Hardshrink::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::hardshrink_impl::cpu
