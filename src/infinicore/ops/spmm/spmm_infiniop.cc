#include "../infiniop_impl.hpp"
#include "infinicore/ops/spmm.hpp"

namespace infinicore::op::spmm_impl::infiniop {

struct Descriptor {
    infiniopSpMMDescriptor_t desc = nullptr;
    SpMat a;

    Descriptor(infiniopSpMMDescriptor_t desc, SpMat a)
        : desc(desc), a(std::move(a)) {}

    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;

    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroySpMMDescriptor(desc);
        }
    }
};

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, b;
    SpMat a;
    float alpha, beta;
};

void *plan(Tensor c, const SpMat &a, const Tensor &b, float alpha, float beta) {
    infiniopSpMMDescriptor_t raw_descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateSpMMDescriptor(
        context::getInfiniopHandle(context::getDevice()),
        &raw_descriptor,
        c->desc(),
        a->desc(),
        b->desc()));
    auto descriptor = std::make_shared<Descriptor>(raw_descriptor, a);

    INFINIOP_WORKSPACE_TENSOR(workspace, SpMM, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(b),
        a,
        alpha,
        beta};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSpMM(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->c->data(), planned->b->data(), planned->alpha, planned->beta, context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SpMM, &plan, &run, &cleanup);

} // namespace infinicore::op::spmm_impl::infiniop
