#include "infinicore/ops/spmv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SpMV);

SpMV::SpMV(Tensor y, const SpMat &a, const Tensor &x, float alpha, float beta) {
    INFINICORE_ASSERT(a);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, a->values(), a->crow_indices(), a->col_indices(), x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, a, x, alpha, beta);
}

void SpMV::execute(Tensor y, const SpMat &a, const Tensor &x, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SpMV, y, a, x, alpha, beta);
}

Tensor spmv(const SpMat &a, const Tensor &x, float alpha, float beta) {
    INFINICORE_ASSERT(a);
    auto y = Tensor::zeros({a->rows()}, a->dtype(), x->device());
    spmv_(y, a, x, alpha, beta);
    return y;
}

void spmv_(Tensor y, const SpMat &a, const Tensor &x, float alpha, float beta) {
    SpMV::execute(y, a, x, alpha, beta);
}

} // namespace infinicore::op
