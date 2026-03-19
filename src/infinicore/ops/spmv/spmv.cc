#include "infinicore/ops/spmv.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<SpMV::schema> &SpMV::dispatcher() {
    static common::OpDispatcher<SpMV::schema> dispatcher_;
    return dispatcher_;
}

void SpMV::execute(
    Tensor output,
    Tensor input_x,
    Tensor values,
    Tensor row_ptr,
    Tensor col_indices,
    size_t rows,
    size_t cols,
    size_t nnzs) {
    
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input_x, values, row_ptr, col_indices);

    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No SpMV implementation for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input_x, values, row_ptr, col_indices, rows, cols, nnzs);
}

Tensor spmv(Tensor input_x, Tensor values, Tensor row_ptr, Tensor col_indices, size_t rows, size_t cols, size_t nnzs) {
    Shape out_shape = {rows};
    auto output = Tensor::empty(out_shape, input_x->dtype(), input_x->device());
    spmv_(output, input_x, values, row_ptr, col_indices, rows, cols, nnzs);
    return output;
}

void spmv_(Tensor output, Tensor input_x, Tensor values, Tensor row_ptr, Tensor col_indices, size_t rows, size_t cols, size_t nnzs) {
    SpMV::execute(output, input_x, values, row_ptr, col_indices, rows, cols, nnzs);
}

} // namespace infinicore::op
