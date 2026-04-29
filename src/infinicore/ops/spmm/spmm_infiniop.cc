#include "../infiniop_impl.hpp"
#include "infinicore/ops/spmm.hpp"

namespace infinicore::op::spmm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Spmm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor c, b;
    graph::GraphTensor row_offsets, col_indices, values;
    float alpha, beta;
};

void *plan(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    size_t seed = hash_combine(c, a.values(), b);
    hash_combine(seed, a.rows());
    hash_combine(seed, a.cols());

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Spmm,
        seed, c->desc(), b->desc(), a.values()->desc(), a.rows(), a.cols());

    INFINIOP_WORKSPACE_TENSOR(workspace, Spmm, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(b),
        graph::GraphTensor(a.row_offsets()),
        graph::GraphTensor(a.col_indices()),
        graph::GraphTensor(a.values()),
        alpha, beta};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSpmm(
        planned->descriptor->desc,
        planned->workspace->data(), planned->workspace->numel(),
        planned->c->data(),
        planned->row_offsets->data(),
        planned->col_indices->data(),
        planned->values->data(),
        planned->b->data(),
        planned->alpha, planned->beta,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Spmm, &plan, &run, &cleanup);

} // namespace infinicore::op::spmm_impl::infiniop
