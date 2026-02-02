#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class CosineEmbeddingLoss {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float, std::string);
    static void execute(Tensor output, Tensor input1, Tensor input2, Tensor target, float margin, std::string reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin = 0.0, std::string reduction = "mean");
void cosine_embedding_loss_(Tensor out, Tensor input1, Tensor input2, Tensor target, float margin = 0.0, std::string reduction = "mean");

} // namespace infinicore::op
