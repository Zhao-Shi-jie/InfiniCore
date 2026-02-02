#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <iterator>

namespace infinicore::op {
class Hardshrink {
public:
    using schema = void (*)(Tensor, Tensor, float);
    static void execute(Tensor output, Tensor input, float lambd);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor hardshrink(Tensor input, float lambd = 0.5f);
void hardshrink_(Tensor output, Tensor input, float lambd = 0.5f);

} // namespace infinicore::op
