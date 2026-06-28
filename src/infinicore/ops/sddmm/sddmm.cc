#include "infinicore/ops/sddmm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SDDMM);

SDDMM::SDDMM(Tensor c_values, SpMat c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_ASSERT(c);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c_values, c->values(), c->crow_indices(), c->col_indices(), a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c_values, c, a, b, alpha, beta);
}

void SDDMM::execute(Tensor c_values, SpMat c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SDDMM, c_values, c, a, b, alpha, beta);
}

void sddmm_(SpMat c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    SDDMM::execute(c->values(), c, a, b, alpha, beta);
}

} // namespace infinicore::op
