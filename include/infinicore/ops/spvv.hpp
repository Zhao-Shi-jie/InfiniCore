#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../spvec.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SpVV, Tensor, const SpVec &, const Tensor &);

Tensor spvv(const SpVec &a, const Tensor &x);
void spvv_(Tensor y, const SpVec &a, const Tensor &x);

} // namespace infinicore::op
