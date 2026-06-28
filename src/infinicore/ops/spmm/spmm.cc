#include "infinicore/ops/spmm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SpMM);

SpMM::SpMM(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_ASSERT(a);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a->values(), a->crow_indices(), a->col_indices(), b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b, alpha, beta);
}

void SpMM::execute(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SpMM, c, a, b, alpha, beta);
}

Tensor spmm(const SpMat &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_ASSERT(a);
    auto c = Tensor::zeros({a->rows(), b->size(1)}, a->dtype(), b->device());
    spmm_(c, a, b, alpha, beta);
    return c;
}

void spmm_(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    SpMM::execute(c, a, b, alpha, beta);
}

} // namespace infinicore::op
