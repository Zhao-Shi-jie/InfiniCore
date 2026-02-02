#include "infinicore/ops/hardsigmoid.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Hardsigmoid::schema> &Hardsigmoid::dispatcher() {
    static common::OpDispatcher<Hardsigmoid::schema> dispatcher_;
    return dispatcher_;
}

void Hardsigmoid::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(output, input);
}

Tensor hardsigmoid(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    hardsigmoid_(output, input);
    return output;
}

void hardsigmoid_(Tensor output, Tensor input) {
    Hardsigmoid::execute(output, input);
}

} // namespace infinicore::op
