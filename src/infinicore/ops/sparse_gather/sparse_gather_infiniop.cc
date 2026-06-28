#include "../infiniop_impl.hpp"
#include "infinicore/ops/sparse_gather.hpp"

namespace infinicore::op::sparse_gather_impl::infiniop {

struct Descriptor {
    infiniopSparseGatherDescriptor_t desc = nullptr;
    SpVec pattern;

    Descriptor(infiniopSparseGatherDescriptor_t desc, SpVec pattern)
        : desc(desc), pattern(std::move(pattern)) {}

    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;

    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroySparseGatherDescriptor(desc);
        }
    }
};

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input;
    SpVec pattern;
};

void *plan(Tensor output, const SpVec &pattern, const Tensor &input) {
    infiniopSparseGatherDescriptor_t raw_descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateSparseGatherDescriptor(
        context::getInfiniopHandle(context::getDevice()),
        &raw_descriptor,
        output->desc(),
        pattern->desc(),
        input->desc(),
        output->data(),
        input->data()));
    auto descriptor = std::make_shared<Descriptor>(raw_descriptor, pattern);

    INFINIOP_WORKSPACE_TENSOR(workspace, SparseGather, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        pattern};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSparseGather(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SparseGather, &plan, &run, &cleanup);

} // namespace infinicore::op::sparse_gather_impl::infiniop
