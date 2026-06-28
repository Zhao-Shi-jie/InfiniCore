#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../spmat.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SDDMM, Tensor, SpMat, const Tensor &, const Tensor &, float, float);

void sddmm_(SpMat c, const Tensor &a, const Tensor &b, float alpha = 1.0f, float beta = 0.0f);

} // namespace infinicore::op
