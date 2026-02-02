#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <iterator>

namespace infinicore::op {
class Hardsigmoid {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor input, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor hardsigmoid(Tensor input);
void hardsigmoid_(Tensor input, Tensor output);
} // namespace infinicore::op
