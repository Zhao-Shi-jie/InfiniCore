#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../spvec.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SparseGather, Tensor, const SpVec &, const Tensor &);

Tensor sparse_gather(const SpVec &pattern, const Tensor &input);
void sparse_gather_(Tensor output, const SpVec &pattern, const Tensor &input);

} // namespace infinicore::op
