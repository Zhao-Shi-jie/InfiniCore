#include "../infiniop_impl.hpp"
#include "infinicore/ops/spvv.hpp"

namespace infinicore::op::spvv_impl::infiniop {

struct Descriptor {
    infiniopSpVVDescriptor_t desc = nullptr;
    SpVec a;

    Descriptor(infiniopSpVVDescriptor_t desc, SpVec a)
        : desc(desc), a(std::move(a)) {}

    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;

    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroySpVVDescriptor(desc);
        }
    }
};

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x;
    SpVec a;
};

void *plan(Tensor y, const SpVec &a, const Tensor &x) {
    infiniopSpVVDescriptor_t raw_descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateSpVVDescriptor(
        context::getInfiniopHandle(context::getDevice()),
        &raw_descriptor,
        y->desc(),
        a->desc(),
        x->desc(),
        x->data()));
    auto descriptor = std::make_shared<Descriptor>(raw_descriptor, a);

    INFINIOP_WORKSPACE_TENSOR(workspace, SpVV, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x),
        a};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSpVV(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->y->data(), planned->x->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SpVV, &plan, &run, &cleanup);

} // namespace infinicore::op::spvv_impl::infiniop
