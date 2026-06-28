#include "infinicore/ops/spvv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SpVV);

SpVV::SpVV(Tensor y, const SpVec &a, const Tensor &x) {
    INFINICORE_ASSERT(a);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, a->values(), a->indices(), x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, a, x);
}

void SpVV::execute(Tensor y, const SpVec &a, const Tensor &x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SpVV, y, a, x);
}

Tensor spvv(const SpVec &a, const Tensor &x) {
    INFINICORE_ASSERT(a);
    auto y = Tensor::empty({}, a->dtype(), x->device());
    spvv_(y, a, x);
    return y;
}

void spvv_(Tensor y, const SpVec &a, const Tensor &x) {
    SpVV::execute(y, a, x);
}

} // namespace infinicore::op
