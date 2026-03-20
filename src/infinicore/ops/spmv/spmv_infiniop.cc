#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/spmv.hpp"
#include <infiniop.h>

namespace infinicore::op::spmv_impl::infiniop {

thread_local common::OpCache<size_t, infiniopSpMVDescriptor_t> caches(
    100,
    [](infiniopSpMVDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySpMVDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(
    Tensor output,
    Tensor input_x,
    Tensor values,
    Tensor row_ptr,
    Tensor col_indices,
    size_t rows,
    size_t cols,
    size_t nnzs) {

    size_t seed = hash_combine(output, input_x, values, row_ptr, col_indices, rows, cols, nnzs);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopSpMVDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSpMVDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            rows,
            cols,
            nnzs));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // size_t workspace_size = 0;
    // INFINICORE_CHECK_ERROR(infiniopGetSpMVWorkspaceSize(desc, &workspace_size));
    // std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopSpMV(
        desc,
        //workspace->data(),
        //workspace_size,
        output->data(),
        input_x->data(),
        values->data(),
        row_ptr->data(),
        col_indices->data(),
        context::getStream()));
}

static bool registered = []() {
    SpMV::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::spmv_impl::infiniop
