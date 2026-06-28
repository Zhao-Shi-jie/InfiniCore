#include "../infiniop_impl.hpp"
#include "infinicore/ops/sparse_scatter.hpp"

namespace infinicore::op::sparse_scatter_impl::infiniop {

struct Descriptor {
    infiniopSparseScatterDescriptor_t desc = nullptr;
    SpVec input;

    Descriptor(infiniopSparseScatterDescriptor_t desc, SpVec input)
        : desc(desc), input(std::move(input)) {}

    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;

    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroySparseScatterDescriptor(desc);
        }
    }
};

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output;
    SpVec input;
};

void *plan(Tensor output, const SpVec &input) {
    infiniopSparseScatterDescriptor_t raw_descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateSparseScatterDescriptor(
        context::getInfiniopHandle(context::getDevice()),
        &raw_descriptor,
        output->desc(),
        input->desc()));
    auto descriptor = std::make_shared<Descriptor>(raw_descriptor, input);

    INFINIOP_WORKSPACE_TENSOR(workspace, SparseScatter, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        input};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSparseScatter(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SparseScatter, &plan, &run, &cleanup);

} // namespace infinicore::op::sparse_scatter_impl::infiniop
