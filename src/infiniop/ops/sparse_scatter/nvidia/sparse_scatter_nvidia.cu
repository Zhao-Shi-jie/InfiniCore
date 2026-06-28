#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "sparse_scatter_nvidia.cuh"

#include <cusparse.h>

namespace op::sparse_scatter::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    cusparseSpVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    cudaDataType data_type = CUDA_R_32F;
    cusparseIndexType_t index_type = CUSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (vec_x != nullptr) {
            cusparseDestroySpVec(vec_x);
        }
        if (vec_y != nullptr) {
            cusparseDestroyDnVec(vec_y);
        }
    }
};

static cudaDataType cudaDataTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        return CUDA_R_32F;
    default:
        return CUDA_R_32F;
    }
}

static cusparseIndexType_t indexTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_I32:
        return CUSPARSE_INDEX_32I;
    case INFINI_DTYPE_I64:
        return CUSPARSE_INDEX_64I;
    default:
        return CUSPARSE_INDEX_64I;
    }
}

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopSpVecDescriptor_t input_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    CHECK_DTYPE(output_desc->dtype(), INFINI_DTYPE_F32);

    auto result = SparseScatterInfo::create(output_desc, input_desc);
    CHECK_RESULT(result);
    CHECK_OR_RETURN(result->output_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto opaque = new Opaque(handle->internal());
    opaque->data_type = cudaDataTypeOf(output_desc->dtype());
    opaque->index_type = indexTypeOf(input_desc->indicesDesc()->dtype());

    auto status = cusparseCreateSpVec(
        &opaque->vec_x,
        static_cast<int64_t>(result->output_vector.size),
        static_cast<int64_t>(result->nnz),
        const_cast<void *>(input_desc->indices()),
        const_cast<void *>(input_desc->values()),
        opaque->index_type,
        CUSPARSE_INDEX_BASE_ZERO,
        opaque->data_type);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    status = cusparseCreateDnVec(
        &opaque->vec_y,
        static_cast<int64_t>(result->output_vector.size),
        const_cast<void *>(input_desc->values()),
        opaque->data_type);
    CHECK_API_OR(status, CUSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    *desc_ptr = new Descriptor(
        output_desc->dtype(),
        input_desc->indicesDesc()->dtype(),
        result.take(),
        input_desc,
        0,
        opaque,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *stream) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_CUSPARSE(cusparseDnVecSetValues(_opaque->vec_y, output));
    CHECK_STATUS(_opaque->internal->useCusparse(
        reinterpret_cast<cudaStream_t>(stream),
        [&](cusparseHandle_t sparse_handle) {
            CHECK_CUSPARSE(cusparseScatter(
                sparse_handle,
                _opaque->vec_x,
                _opaque->vec_y));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::sparse_scatter::nvidia
