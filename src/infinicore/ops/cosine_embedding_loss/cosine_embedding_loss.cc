#include "infinicore/ops/cosine_embedding_loss.hpp"
#include "../../utils.hpp"
#include <iterator>

namespace infinicore::op {

common::OpDispatcher<CosineEmbeddingLoss::schema> &CosineEmbeddingLoss::dispatcher() {
    static common::OpDispatcher<CosineEmbeddingLoss::schema> dispatcher_;
    return dispatcher_;
}

void CosineEmbeddingLoss::execute(Tensor output, Tensor input1, Tensor input2, Tensor target, float margin, std::string reduction) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input1, input2, target);
    infinicore::context::setDevice(input1->device());
    dispatcher().lookup(input1->device().getType())(output, input1, input2, target, margin, reduction);
}

Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin, std::string reduction) {
    auto input_shape = input1->shape();

    infinicore::Shape output_shape;
    if (reduction == "none") {
        if (input_shape.size() > 1) {
            output_shape = infinicore::Shape(input_shape.begin(), input_shape.end() - 1);
        } else {
            output_shape = infinicore::Shape({1});
        }
    } else {
        output_shape = infinicore::Shape({1});
    }

    auto output = Tensor::empty(output_shape, input1->dtype(), input1->device());
    cosine_embedding_loss_(output, input1, input2, target, margin, reduction);
    return output;
}

void cosine_embedding_loss_(Tensor output, Tensor input1, Tensor input2, Tensor target, float margin, std::string reduction) {
    CosineEmbeddingLoss::execute(output, input1, input2, target, margin, reduction);
}

} // namespace infinicore::op
