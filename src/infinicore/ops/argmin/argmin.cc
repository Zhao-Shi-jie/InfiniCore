#include "infinicore/ops/argmin.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Argmin::schema> &Argmin::dispatcher() {
    static common::OpDispatcher<Argmin::schema> dispatcher_;
    return dispatcher_;
}

void Argmin::execute(Tensor output, Tensor input, int dim, bool keepdim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(output, input, dim, keepdim);
}

Tensor argmin(Tensor input, int dim, bool keepdim) {
    auto input_shape = input->shape();
    infinicore::Shape output_shape = input_shape;
    if (keepdim) {
        output_shape[dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + dim);
    }
    auto output = Tensor::empty(output_shape, DataType::I64, input->device());
    argmin_(output, input, dim, keepdim);
    return output;
}

void argmin_(Tensor output, Tensor input, int dim, bool keepdim) {
    Argmin::execute(output, input, dim, keepdim);
}

} // namespace infinicore::op
