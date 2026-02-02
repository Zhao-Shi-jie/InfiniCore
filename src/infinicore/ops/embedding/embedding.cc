#include "infinicore/ops/embedding.hpp"

namespace infinicore::op {

common::OpDispatcher<Embedding::schema> &Embedding::dispatcher() {
    static common::OpDispatcher<Embedding::schema> dispatcher_;
    return dispatcher_;
}

void Embedding::execute(Tensor out, Tensor input, Tensor weight) {
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(out, input, weight);
}

Tensor embedding(Tensor input, Tensor weight) {
    auto input_shape = input->shape();
    auto weight_shape = weight->shape();
    auto embedding_dim = weight_shape[1];

    // Assign memory to out variables
    auto output_shape = input_shape;
    output_shape.push_back(embedding_dim);
    auto inputs_embeds = Tensor::empty(output_shape, weight->dtype(), weight->device());

    embedding_(inputs_embeds, input, weight);
    return inputs_embeds;
}

void embedding_(Tensor out, Tensor input, Tensor weight) {
    Embedding::execute(out, input, weight);
}

} // namespace infinicore::op
