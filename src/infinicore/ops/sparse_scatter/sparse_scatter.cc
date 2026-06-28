#include "infinicore/ops/sparse_scatter.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SparseScatter);

SparseScatter::SparseScatter(Tensor output, const SpVec &input) {
    INFINICORE_ASSERT(input);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input->values(), input->indices());
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input);
}

void SparseScatter::execute(Tensor output, const SpVec &input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SparseScatter, output, input);
}

Tensor sparse_scatter(const SpVec &input) {
    INFINICORE_ASSERT(input);
    auto output = Tensor::zeros({input->size()}, input->dtype(), input->device());
    sparse_scatter_(output, input);
    return output;
}

void sparse_scatter_(Tensor output, const SpVec &input) {
    SparseScatter::execute(output, input);
}

} // namespace infinicore::op
