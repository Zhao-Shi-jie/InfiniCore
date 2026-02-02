#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/argmin.hpp"
#include <omp.h>
#include <vector>

namespace infinicore::op::argmin_impl::cpu {

template <typename T>
void argmin_reduce_contiguous(const T *input_data, int64_t *output_data,
                              const std::vector<size_t> &input_shape,
                              int dim, bool keepdim) {
    int ndim = input_shape.size();
    size_t dim_size = input_shape[dim];

    size_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= input_shape[i];
    }
    size_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= input_shape[i];
    }

    size_t output_numel = outer_size * inner_size;

#pragma omp parallel for
    for (size_t out_idx = 0; out_idx < output_numel; ++out_idx) {
        size_t outer_idx = out_idx / inner_size;
        size_t inner_idx = out_idx % inner_size;

        size_t min_idx = 0;
        T min_val = input_data[outer_idx * dim_size * inner_size + 0 * inner_size + inner_idx];

        for (size_t d = 1; d < dim_size; ++d) {
            T val = input_data[outer_idx * dim_size * inner_size + d * inner_size + inner_idx];
            if (utils::cast<float>(val) < utils::cast<float>(min_val)) {
                min_val = val;
                min_idx = d;
            }
        }

        output_data[out_idx] = static_cast<int64_t>(min_idx);
    }
}

template <typename T>
void argmin_reduce_strided(const T *input_base, int64_t *output_base,
                           const std::vector<size_t> &input_shape,
                           const std::vector<int64_t> &input_strides,
                           const std::vector<size_t> &output_shape,
                           const std::vector<int64_t> &output_strides,
                           int dim, bool keepdim) {
    int ndim = input_shape.size();
    size_t dim_size = input_shape[dim];

    size_t output_numel = 1;
    for (auto s : output_shape) {
        output_numel *= s;
    }

#pragma omp parallel for
    for (size_t out_linear_idx = 0; out_linear_idx < output_numel; ++out_linear_idx) {
        size_t temp_idx = out_linear_idx;
        std::vector<size_t> out_coords(ndim - (keepdim ? 0 : 1));

        int out_dim_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == dim && !keepdim) {
                continue;
            }
            out_coords[out_dim_idx] = temp_idx % output_shape[out_dim_idx];
            temp_idx /= output_shape[out_dim_idx];
            ++out_dim_idx;
        }

        size_t min_idx = 0;
        size_t input_offset = 0;

        int coord_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == dim) {
                input_offset += 0 * input_strides[d]; // dim 维度设为 0
            } else {
                input_offset += out_coords[coord_idx] * input_strides[d];
                ++coord_idx;
            }
        }
        T min_val = input_base[input_offset];

        for (size_t d = 1; d < dim_size; ++d) {
            input_offset = 0;
            coord_idx = 0;
            for (int dd = 0; dd < ndim; ++dd) {
                if (dd == dim) {
                    input_offset += d * input_strides[dd];
                } else {
                    input_offset += out_coords[coord_idx] * input_strides[dd];
                    ++coord_idx;
                }
            }
            T val = input_base[input_offset];
            if (utils::cast<float>(val) < utils::cast<float>(min_val)) {
                min_val = val;
                min_idx = d;
            }
        }

        size_t output_offset = 0;
        coord_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == dim && !keepdim) {
                continue;
            }
            output_offset += out_coords[coord_idx] * output_strides[coord_idx];
            ++coord_idx;
        }

        output_base[output_offset] = static_cast<int64_t>(min_idx);
    }
}

void calculate(Tensor output, Tensor input, int dim, bool keepdim) {
    auto ndim = input->ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Invalid dimension for argmin");
    }

    auto dtype = input->dtype();
    auto input_shape = input->shape();
    auto input_strides = input->strides();

    auto output_shape = output->shape();
    auto output_strides = output->strides();

    bool input_contiguous = input->is_contiguous();
    bool output_contiguous = output->is_contiguous();

    auto input_base = input->data();
    auto output_base = reinterpret_cast<int64_t *>(output->data());

    if (dtype == DataType::F32) {
        const auto *in_ptr = reinterpret_cast<const float *>(input_base);
        if (input_contiguous && output_contiguous) {
            argmin_reduce_contiguous<float>(in_ptr, output_base, input_shape, dim, keepdim);
        } else {
            argmin_reduce_strided<float>(in_ptr, output_base, input_shape, input_strides,
                                         output_shape, output_strides, dim, keepdim);
        }
    } else if (dtype == DataType::F64) {
        const auto *in_ptr = reinterpret_cast<const double *>(input_base);
        if (input_contiguous && output_contiguous) {
            argmin_reduce_contiguous<double>(in_ptr, output_base, input_shape, dim, keepdim);
        } else {
            argmin_reduce_strided<double>(in_ptr, output_base, input_shape, input_strides,
                                          output_shape, output_strides, dim, keepdim);
        }
    } else if (dtype == DataType::F16) {
        const auto *in_ptr = reinterpret_cast<const fp16_t *>(input_base);
        if (input_contiguous && output_contiguous) {
            argmin_reduce_contiguous<fp16_t>(in_ptr, output_base, input_shape, dim, keepdim);
        } else {
            argmin_reduce_strided<fp16_t>(in_ptr, output_base, input_shape, input_strides,
                                          output_shape, output_strides, dim, keepdim);
        }
    } else if (dtype == DataType::BF16) {
        const auto *in_ptr = reinterpret_cast<const bf16_t *>(input_base);
        if (input_contiguous && output_contiguous) {
            argmin_reduce_contiguous<bf16_t>(in_ptr, output_base, input_shape, dim, keepdim);
        } else {
            argmin_reduce_strided<bf16_t>(in_ptr, output_base, input_shape, input_strides,
                                          output_shape, output_strides, dim, keepdim);
        }
    } else if (dtype == DataType::I32) {
        const auto *in_ptr = reinterpret_cast<const int32_t *>(input_base);
        if (input_contiguous && output_contiguous) {
            argmin_reduce_contiguous<int32_t>(in_ptr, output_base, input_shape, dim, keepdim);
        } else {
            argmin_reduce_strided<int32_t>(in_ptr, output_base, input_shape, input_strides,
                                           output_shape, output_strides, dim, keepdim);
        }
    } else {
        throw std::runtime_error("Unsupported dtype for argmin.");
    }
}

static bool registered = []() {
    Argmin::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::argmin_impl::cpu
