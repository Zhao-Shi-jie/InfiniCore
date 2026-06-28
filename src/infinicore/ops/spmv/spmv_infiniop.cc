#include "../infiniop_impl.hpp"
#include "infinicore/ops/spmv.hpp"

namespace infinicore::op::spmv_impl::infiniop {

struct Descriptor {
    infiniopSpMVDescriptor_t desc = nullptr;
    SpMat a;

    Descriptor(infiniopSpMVDescriptor_t desc, SpMat a)
        : desc(desc), a(std::move(a)) {}

    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;

    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroySpMVDescriptor(desc);
        }
    }
};

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x;
    SpMat a;
    float alpha, beta;
};

void *plan(Tensor y, const SpMat &a, const Tensor &x, float alpha, float beta) {
    infiniopSpMVDescriptor_t raw_descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateSpMVDescriptor(
        context::getInfiniopHandle(context::getDevice()),
        &raw_descriptor,
        y->desc(),
        a->desc(),
        x->desc(),
        y->data(),
        x->data()));
    auto descriptor = std::make_shared<Descriptor>(raw_descriptor, a);

    INFINIOP_WORKSPACE_TENSOR(workspace, SpMV, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x),
        a,
        alpha,
        beta};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSpMV(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->y->data(), planned->x->data(), planned->alpha, planned->beta, context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SpMV, &plan, &run, &cleanup);

} // namespace infinicore::op::spmv_impl::infiniop
