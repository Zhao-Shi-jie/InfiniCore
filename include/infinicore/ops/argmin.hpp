#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <iterator>

namespace infinicore::op {
class Argmin {
public:
    using schema = void (*)(Tensor, Tensor, int, bool);
    static void execute(Tensor output, Tensor input, int dim, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor argmin(Tensor input, int dim, bool keepdim = false);
void argmin_(Tensor output, Tensor input, int dim, bool keepdim = false);
} // namespace infinicore::op
