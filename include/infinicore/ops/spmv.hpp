#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class SpMV {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, size_t, size_t, size_t);
    static void execute(Tensor output, Tensor input_x, Tensor values, Tensor row_ptr, Tensor col_indices, size_t rows, size_t cols, size_t nnzs);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor spmv(Tensor input_x, Tensor values, Tensor row_ptr, Tensor col_indices, size_t rows, size_t cols, size_t nnzs);
void spmv_(Tensor output, Tensor input_x, Tensor values, Tensor row_ptr, Tensor col_indices, size_t rows, size_t cols, size_t nnzs);

} // namespace infinicore::op
