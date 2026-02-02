#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/cosine_embedding_loss.hpp"
#include <cmath>
#include <omp.h>
#include <string>
#include <vector>

namespace infinicore::op::cosine_embedding_loss_impl::cpu {

template <typename T>
float dot_product(const T *a, const T *b, size_t size,
                  int64_t stride_a, int64_t stride_b) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += utils::cast<float>(a[i * stride_a]) * utils::cast<float>(b[i * stride_b]);
    }
    return sum;
}

template <typename T>
float vector_norm(const T *a, size_t size, int64_t stride) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float val = utils::cast<float>(a[i * stride]);
        sum += val * val;
    }
    return std::sqrt(sum);
}

template <typename T>
float cosine_similarity(const T *a, const T *b, size_t size,
                        int64_t stride_a, int64_t stride_b) {
    float dot = dot_product(a, b, size, stride_a, stride_b);
    float norm_a = vector_norm(a, size, stride_a);
    float norm_b = vector_norm(b, size, stride_b);
    float epsilon = 1e-8f;
    return dot / (norm_a * norm_b + epsilon);
}

template <typename T>
void cosine_embedding_loss_contiguous(
    T *output_data,
    const T *input1_data,
    const T *input2_data,
    const T *target_data,
    size_t batch_size,
    size_t embedding_dim,
    float margin) {

#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        const T *x1 = input1_data + b * embedding_dim;
        const T *x2 = input2_data + b * embedding_dim;
        float y_val = utils::cast<float>(target_data[b]);

        float cos_sim = cosine_similarity(x1, x2, embedding_dim, 1, 1);

        float loss;
        if (y_val > 0.0f) {
            loss = 1.0f - cos_sim;
        } else {
            loss = std::max(0.0f, cos_sim - margin);
        }

        output_data[b] = utils::cast<T>(loss);
    }
}

template <typename T>
void cosine_embedding_loss_strided(
    T *output_base,
    const T *input1_base,
    const T *input2_base,
    const T *target_base,
    const std::vector<size_t> &input_shape,
    const std::vector<int64_t> &input1_strides,
    const std::vector<int64_t> &input2_strides,
    const std::vector<int64_t> &target_strides,
    const std::vector<int64_t> &output_strides,
    float margin) {

    int ndim = input_shape.size();
    size_t embedding_dim = input_shape[ndim - 1];

    size_t batch_size = 1;
    for (int i = 0; i < ndim - 1; ++i) {
        batch_size *= input_shape[i];
    }

    std::vector<size_t> batch_shape(input_shape.begin(), input_shape.end() - 1);

#pragma omp parallel for
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t temp_idx = batch_idx;
        std::vector<size_t> coords(ndim - 1);
        for (int d = ndim - 2; d >= 0; --d) {
            coords[d] = temp_idx % batch_shape[d];
            temp_idx /= batch_shape[d];
        }

        size_t input1_offset = 0;
        size_t input2_offset = 0;
        size_t target_offset = 0;
        size_t output_offset = 0;

        for (int d = 0; d < ndim - 1; ++d) {
            input1_offset += coords[d] * input1_strides[d];
            input2_offset += coords[d] * input2_strides[d];
            target_offset += coords[d] * target_strides[d];
            output_offset += coords[d] * output_strides[d];
        }

        const T *x1 = input1_base + input1_offset;
        const T *x2 = input2_base + input2_offset;
        float y_val = utils::cast<float>(target_base[target_offset]);

        float cos_sim = cosine_similarity(x1, x2, embedding_dim,
                                          input1_strides[ndim - 1],
                                          input2_strides[ndim - 1]);

        float loss;
        if (y_val > 0.0f) {
            loss = 1.0f - cos_sim;
        } else {
            loss = std::max(0.0f, cos_sim - margin);
        }

        output_base[output_offset] = utils::cast<T>(loss);
    }
}

template <typename T>
float apply_reduction(const T *data, size_t size, const std::string &reduction) {
    float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < size; ++i) {
        sum += utils::cast<float>(data[i]);
    }

    if (reduction == "mean") {
        return sum / static_cast<float>(size);
    } else { // "sum"
        return sum;
    }
}

template <typename T>
void cosine_embedding_loss_contiguous_from_strided(
    T *output_data,
    const T *input1_base,
    const T *input2_base,
    const T *target_base,
    const std::vector<size_t> &input_shape,
    const std::vector<int64_t> &input1_strides,
    const std::vector<int64_t> &input2_strides,
    const std::vector<int64_t> &target_strides,
    size_t batch_size,
    size_t embedding_dim,
    float margin) {

    int ndim = input_shape.size();
    std::vector<size_t> batch_shape(input_shape.begin(), input_shape.end() - 1);

#pragma omp parallel for
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t temp_idx = batch_idx;
        std::vector<size_t> coords(ndim - 1);
        for (int d = ndim - 2; d >= 0; --d) {
            coords[d] = temp_idx % batch_shape[d];
            temp_idx /= batch_shape[d];
        }

        size_t input1_offset = 0;
        size_t input2_offset = 0;
        size_t target_offset = 0;

        for (int d = 0; d < ndim - 1; ++d) {
            input1_offset += coords[d] * input1_strides[d];
            input2_offset += coords[d] * input2_strides[d];
            target_offset += coords[d] * target_strides[d];
        }

        const T *x1 = input1_base + input1_offset;
        const T *x2 = input2_base + input2_offset;
        float y_val = utils::cast<float>(target_base[target_offset]);

        float cos_sim = cosine_similarity(x1, x2, embedding_dim,
                                          input1_strides[ndim - 1],
                                          input2_strides[ndim - 1]);

        float loss;
        if (y_val > 0.0f) {
            loss = 1.0f - cos_sim;
        } else {
            loss = std::max(0.0f, cos_sim - margin);
        }

        output_data[batch_idx] = utils::cast<T>(loss);
    }
}

template <typename T>
void calculate_typed(Tensor output, Tensor input1, Tensor input2, Tensor target,
                     float margin, const std::string &reduction) {
    auto input_shape = input1->shape();
    int ndim = input_shape.size();

    size_t embedding_dim = input_shape[ndim - 1];
    size_t batch_size = 1;
    for (int i = 0; i < ndim - 1; ++i) {
        batch_size *= input_shape[i];
    }

    bool all_contiguous = input1->is_contiguous() && input2->is_contiguous()
                       && target->is_contiguous() && output->is_contiguous();

    auto *in1_ptr = reinterpret_cast<const T *>(input1->data());
    auto *in2_ptr = reinterpret_cast<const T *>(input2->data());
    auto *target_ptr = reinterpret_cast<const T *>(target->data());
    auto *output_ptr = reinterpret_cast<T *>(output->data());

    if (reduction == "none") {
        if (all_contiguous) {
            cosine_embedding_loss_contiguous<T>(output_ptr, in1_ptr, in2_ptr,
                                                target_ptr, batch_size,
                                                embedding_dim, margin);
        } else {
            cosine_embedding_loss_strided<T>(output_ptr, in1_ptr, in2_ptr, target_ptr,
                                             input_shape, input1->strides(),
                                             input2->strides(), target->strides(),
                                             output->strides(), margin);
        }
    } else {
        std::vector<T> per_sample_loss(batch_size);
        T *loss_ptr = per_sample_loss.data();

        if (all_contiguous) {
            cosine_embedding_loss_contiguous<T>(loss_ptr, in1_ptr, in2_ptr,
                                                target_ptr, batch_size,
                                                embedding_dim, margin);
        } else {
            std::vector<size_t> batch_shape(input_shape.begin(), input_shape.end() - 1);
            cosine_embedding_loss_contiguous_from_strided<T>(
                loss_ptr, in1_ptr, in2_ptr, target_ptr,
                input_shape, input1->strides(), input2->strides(), target->strides(),
                batch_size, embedding_dim, margin);
        }

        float reduced = apply_reduction<T>(loss_ptr, batch_size, reduction);
        output_ptr[0] = utils::cast<T>(reduced);
    }
}

void calculate(Tensor output, Tensor input1, Tensor input2, Tensor target,
               float margin, std::string reduction) {
    auto dtype = input1->dtype();
    auto input_shape = input1->shape();

    if (input_shape.size() < 1) {
        throw std::runtime_error("Input tensors must have at least 1 dimension");
    }

    if (dtype == DataType::F32) {
        calculate_typed<float>(output, input1, input2, target, margin, reduction);
    } else if (dtype == DataType::F64) {
        calculate_typed<double>(output, input1, input2, target, margin, reduction);
    } else if (dtype == DataType::F16) {
        calculate_typed<fp16_t>(output, input1, input2, target, margin, reduction);
    } else if (dtype == DataType::BF16) {
        calculate_typed<bf16_t>(output, input1, input2, target, margin, reduction);
    } else {
        throw std::runtime_error("Unsupported dtype for cosine_embedding_loss.");
    }
}

static bool registered = []() {
    CosineEmbeddingLoss::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::cosine_embedding_loss_impl::cpu
