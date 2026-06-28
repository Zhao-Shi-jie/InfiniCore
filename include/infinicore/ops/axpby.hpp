#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../spvec.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Axpby, const SpVec &, Tensor, float, float);

Tensor axpby(const SpVec &x, const Tensor &y, float alpha = 1.0f, float beta = 1.0f);
void axpby_(const SpVec &x, Tensor y, float alpha, float beta);

} // namespace infinicore::op
