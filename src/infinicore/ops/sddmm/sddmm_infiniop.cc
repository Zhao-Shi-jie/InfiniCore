#include "infinicore/ops/sddmm.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::sddmm_impl::infiniop {

struct Descriptor {
    infiniopSDDMMDescriptor_t desc = nullptr;
    SpMat c;

    Descriptor(infiniopSDDMMDescriptor_t desc, SpMat c)
        : desc(desc), c(std::move(c)) {}

    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;

    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroySDDMMDescriptor(desc);
        }
    }
};

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c_values, a, b;
    SpMat c;
    float alpha, beta;
};

void *plan(Tensor c_values, SpMat c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    infiniopSDDMMDescriptor_t raw_descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateSDDMMDescriptor(
        context::getInfiniopHandle(context::getDevice()),
        &raw_descriptor,
        c->desc(),
        a->desc(),
        b->desc()));
    auto descriptor = std::make_shared<Descriptor>(raw_descriptor, c);

    INFINIOP_WORKSPACE_TENSOR(workspace, SDDMM, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c_values),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        c,
        alpha,
        beta};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSDDMM(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->c_values->data(),
        planned->a->data(),
        planned->b->data(),
        planned->alpha,
        planned->beta,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SDDMM, &plan, &run, &cleanup);

} // namespace infinicore::op::sddmm_impl::infiniop
