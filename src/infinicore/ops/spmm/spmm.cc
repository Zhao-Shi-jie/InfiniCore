#include "infinicore/ops/spmm.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Spmm);

Spmm::Spmm(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_GRAPH_OP_DISPATCH(a.device().getType(), c, a, b, alpha, beta);
}

void Spmm::execute(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Spmm, c, a, b, alpha, beta);
}

Tensor spmm(const SpMat &a, const Tensor &b, float alpha) {
    auto c = Tensor::zeros({a.rows(), b->size(1)}, a.dtype(), a.device());
    spmm_(c, a, b, alpha, 0.0f);
    return c;
}

void spmm_(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_ASSERT(a.device() == b->device());
    INFINICORE_ASSERT(a.device() == c->device());
    Spmm::execute(c, a, b, alpha, beta);
}

} // namespace infinicore::op
