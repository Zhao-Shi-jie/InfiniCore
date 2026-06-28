#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../spmat.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SpMV, Tensor, const SpMat &, const Tensor &, float, float);

Tensor spmv(const SpMat &a, const Tensor &x, float alpha = 1.0f, float beta = 0.0f);
void spmv_(Tensor y, const SpMat &a, const Tensor &x, float alpha, float beta);

} // namespace infinicore::op
