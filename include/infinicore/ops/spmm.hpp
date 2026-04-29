#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../sparse.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Spmm, Tensor, const SpMat &, const Tensor &, float, float);

Tensor spmm(const SpMat &a, const Tensor &b, float alpha = 1.0f);
void spmm_(Tensor c, const SpMat &a, const Tensor &b,
           float alpha = 1.0f, float beta = 0.0f);

} // namespace infinicore::op
