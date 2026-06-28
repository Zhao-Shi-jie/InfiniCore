#include "infinicore/ops/axpby.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::axpby_impl::infiniop {

struct Descriptor {
    infiniopAxpbyDescriptor_t desc = nullptr;
    SpVec x;

    Descriptor(infiniopAxpbyDescriptor_t desc, SpVec x)
        : desc(desc), x(std::move(x)) {}

    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;

    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroyAxpbyDescriptor(desc);
        }
    }
};

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y;
    SpVec x;
    float alpha, beta;
};

void *plan(const SpVec &x, Tensor y, float alpha, float beta) {
    infiniopAxpbyDescriptor_t raw_descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateAxpbyDescriptor(
        context::getInfiniopHandle(context::getDevice()),
        &raw_descriptor,
        x->desc(),
        y->desc()));
    auto descriptor = std::make_shared<Descriptor>(raw_descriptor, x);
    INFINIOP_WORKSPACE_TENSOR(workspace, Axpby, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        x,
        alpha,
        beta};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopAxpby(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->y->data(),
        planned->alpha,
        planned->beta,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Axpby, &plan, &run, &cleanup);

} // namespace infinicore::op::axpby_impl::infiniop
