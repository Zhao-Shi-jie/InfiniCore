#include "infinicore/ops/hardshrink.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Hardshrink::schema> &Hardshrink::dispatcher() {
    static common::OpDispatcher<Hardshrink::schema> dispatcher_;
    return dispatcher_;
}

void Hardshrink::execute(Tensor output, Tensor input, float lambd) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(output, input, lambd);
}

Tensor hardshrink(Tensor input, float lambd) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    hardshrink_(output, input, lambd);
    return output;
}

void hardshrink_(Tensor output, Tensor input, float lambd) {
    Hardshrink::execute(output, input, lambd);
}

} // namespace infinicore::op
