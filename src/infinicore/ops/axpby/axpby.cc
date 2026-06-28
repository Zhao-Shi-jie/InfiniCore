#include "infinicore/ops/axpby.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Axpby);

Axpby::Axpby(const SpVec &x, Tensor y, float alpha, float beta) {
    INFINICORE_ASSERT(x);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x->values(), x->indices(), y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), x, y, alpha, beta);
}

void Axpby::execute(const SpVec &x, Tensor y, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Axpby, x, y, alpha, beta);
}

Tensor axpby(const SpVec &x, const Tensor &y, float alpha, float beta) {
    auto out = Tensor::empty(y->shape(), y->dtype(), y->device());
    out->copy_from(y);
    axpby_(x, out, alpha, beta);
    return out;
}

void axpby_(const SpVec &x, Tensor y, float alpha, float beta) {
    Axpby::execute(x, y, alpha, beta);
}

} // namespace infinicore::op
