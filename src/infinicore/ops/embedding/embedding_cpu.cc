#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/embedding.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

namespace infinicore::op::embedding_impl::cpu {

void calculate(Tensor out, Tensor input, Tensor weight) {
    assert(infinicore::DataType::I64 == input->dtype() || (infinicore::DataType::I32 == input->dtype()));

    auto input_shape = input->shape();
    auto weight_shape = weight->shape();
    auto embedding_dim = weight_shape[1];

    Size counts = 1;
    for (auto &v : input_shape) {
        counts *= v;
    }

    const Size bytes = dsize(weight->dtype()) * embedding_dim;
    auto *weight_ptr = weight->data();
    auto *out_ptr = out->data();

    if (weight->device().getType() == Device::Type::CPU) {
        if (infinicore::DataType::I64 == input->dtype()) {
            const int64_t *input_arr = reinterpret_cast<const int64_t *>(input->data());
            for (Size i = 0; i < counts; ++i) {
                int64_t idx = input_arr[i];
                assert((idx >= 0) && (idx < weight_shape[0]));
                std::memcpy(out_ptr + i * bytes, weight_ptr + idx * bytes, bytes);
            }
        } else if (infinicore::DataType::I32 == input->dtype()) {
            const int32_t *input_arr = reinterpret_cast<const int32_t *>(input->data());
            for (Size i = 0; i < counts; ++i) {
                int32_t idx = input_arr[i];
                assert((idx >= 0) && (idx < weight_shape[0]));
                std::memcpy(out_ptr + i * bytes, weight_ptr + idx * bytes, bytes);
            }
        }
    } else {
        if (infinicore::DataType::I64 == input->dtype()) {
            const int64_t *input_arr = reinterpret_cast<const int64_t *>(input->data());
            for (Size i = 0; i < counts; ++i) {
                int64_t idx = input_arr[i];
                assert((idx >= 0) && (idx < weight_shape[0]));
                context::memcpyD2D(out_ptr + i * bytes, weight_ptr + idx * bytes, bytes);
            }
        } else if (infinicore::DataType::I32 == input->dtype()) {
            const int32_t *input_arr = reinterpret_cast<const int32_t *>(input->data());
            for (Size i = 0; i < counts; ++i) {
                int32_t idx = input_arr[i];
                assert((idx >= 0) && (idx < weight_shape[0]));
                context::memcpyD2D(out_ptr + i * bytes, weight_ptr + idx * bytes, bytes);
            }
        }
    }
}

static bool registered = []() {
    Embedding::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::embedding_impl::cpu
