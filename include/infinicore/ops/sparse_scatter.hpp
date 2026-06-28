#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../spvec.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SparseScatter, Tensor, const SpVec &);

Tensor sparse_scatter(const SpVec &input);
void sparse_scatter_(Tensor output, const SpVec &input);

} // namespace infinicore::op
