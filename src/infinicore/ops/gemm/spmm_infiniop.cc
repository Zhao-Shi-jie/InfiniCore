#include "../infiniop_impl.hpp"
#include "infinicore/ops/gemm.hpp"

namespace infinicore::op::spmm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, SpMM, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, b;
    SpMat a;
    float alpha, beta;
};

void *plan(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    size_t seed = hash_combine(c, a->values(), a->crow_indices(), a->col_indices(), b);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, SpMM,
        seed, c->desc(), a->desc(), b->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, SpMM, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(b),
        a,
        alpha, beta};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSpMM(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->c->data(),
        planned->b->data(),
        planned->alpha,
        planned->beta,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SpMM, &plan, &run, &cleanup);

} // namespace infinicore::op::spmm_impl::infiniop
