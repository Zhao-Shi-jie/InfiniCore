#include "infinicore/ops/sparse_gather.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SparseGather);

SparseGather::SparseGather(Tensor output, const SpVec &pattern, const Tensor &input) {
    INFINICORE_ASSERT(pattern);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, pattern->values(), pattern->indices(), input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, pattern, input);
}

void SparseGather::execute(Tensor output, const SpVec &pattern, const Tensor &input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SparseGather, output, pattern, input);
}

Tensor sparse_gather(const SpVec &pattern, const Tensor &input) {
    INFINICORE_ASSERT(pattern);
    auto output = Tensor::empty({pattern->nnz()}, pattern->dtype(), input->device());
    sparse_gather_(output, pattern, input);
    return output;
}

void sparse_gather_(Tensor output, const SpVec &pattern, const Tensor &input) {
    SparseGather::execute(output, pattern, input);
}

} // namespace infinicore::op
