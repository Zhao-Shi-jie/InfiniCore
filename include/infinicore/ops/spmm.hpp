#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../spmat.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SpMM, Tensor, const SpMat &, const Tensor &, float, float);

Tensor spmm(const SpMat &a, const Tensor &b, float alpha = 1.0f, float beta = 0.0f);
void spmm_(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta);

} // namespace infinicore::op
