#include "sparse_scatter_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"
#include "../../../../utils.h"

#include <cstdint>

namespace op::sparse_scatter::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    hpccDataType data_type = HPCC_R_32F;
    hcsparseIndexType_t index_type = HCSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::metax::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() = default;
};

static hpccDataType dataTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        return HPCC_R_16F;
    case INFINI_DTYPE_BF16:
        return HPCC_R_16BF;
    case INFINI_DTYPE_F32:
        return HPCC_R_32F;
    case INFINI_DTYPE_F64:
        return HPCC_R_64F;
    default:
        return HPCC_R_32F;
    }
}

static hcsparseIndexType_t indexTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_I32:
        return HCSPARSE_INDEX_32I;
    case INFINI_DTYPE_I64:
        return HCSPARSE_INDEX_64I;
    default:
        return HCSPARSE_INDEX_64I;
    }
}

static bool isAligned16(const void *ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xf) == 0;
}

static size_t calculateWorkspaceSize(
    infiniopSpVecDescriptor_t input_desc,
    size_t nnz,
    infiniDtype_t dtype,
    infiniDtype_t index_dtype) {
    size_t size = 0;
    if (!isAligned16(input_desc->values())) {
        size = utils::align(size, 16);
        size += nnz * infiniSizeOf(dtype);
    }
    if (!isAligned16(input_desc->indices())) {
        size = utils::align(size, 16);
        size += nnz * infiniSizeOf(index_dtype);
    }
    return size;
}

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopSpVecDescriptor_t input_desc) {
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = SparseScatterInfo::create(output_desc, input_desc);
    CHECK_RESULT(result);
    CHECK_OR_RETURN(result->output_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto opaque = new Opaque(handle->internal());
    opaque->data_type = dataTypeOf(output_desc->dtype());
    opaque->index_type = indexTypeOf(input_desc->indicesDesc()->dtype());
    auto workspace_size = calculateWorkspaceSize(
        input_desc,
        result->nnz,
        output_desc->dtype(),
        input_desc->indicesDesc()->dtype());

    *desc_ptr = new Descriptor(
        output_desc->dtype(),
        input_desc->indicesDesc()->dtype(),
        result.take(),
        input_desc,
        workspace_size,
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

    auto values = const_cast<void *>(_input_desc->values());
    auto indices = const_cast<void *>(_input_desc->indices());
    auto stream_ = reinterpret_cast<hcStream_t>(stream);

    auto base = reinterpret_cast<char *>(workspace);
    size_t offset = 0;
    auto values_bytes = _info.nnz * infiniSizeOf(_dtype);
    auto indices_bytes = _info.nnz * infiniSizeOf(_index_dtype);

    if (!isAligned16(values)) {
        offset = utils::align(offset, 16);
        values = base + offset;
        offset += values_bytes;
        CHECK_INTERNAL(hcMemcpyAsync(
                           values,
                           _input_desc->values(),
                           values_bytes,
                           hcMemcpyDeviceToDevice,
                           stream_),
                       hcSuccess);
    }
    if (!isAligned16(indices)) {
        offset = utils::align(offset, 16);
        indices = base + offset;
        offset += indices_bytes;
        CHECK_INTERNAL(hcMemcpyAsync(
                           indices,
                           _input_desc->indices(),
                           indices_bytes,
                           hcMemcpyDeviceToDevice,
                           stream_),
                       hcSuccess);
    }

    hcsparseSpVecDescr_t vec_x = nullptr;
    auto status = hcsparseCreateSpVec(
        &vec_x,
        static_cast<int64_t>(_info.output_vector.size),
        static_cast<int64_t>(_info.nnz),
        indices,
        values,
        _opaque->index_type,
        HCSPARSE_INDEX_BASE_ZERO,
        _opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, return INFINI_STATUS_INTERNAL_ERROR);

    hcsparseDnVecDescr_t vec_y = nullptr;
    status = hcsparseCreateDnVec(
        &vec_y,
        static_cast<int64_t>(_info.output_vector.size),
        output,
        _opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        hcsparseDestroySpVec(vec_x);
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    auto ret = _opaque->internal->useMcsparse(
        reinterpret_cast<hcStream_t>(stream),
        [&](hcsparseHandle_t sparse_handle) {
            CHECK_MCSPARSE(hcsparseScatter(
                sparse_handle,
                vec_x,
                vec_y));
            return INFINI_STATUS_SUCCESS;
        });
    CHECK_MCSPARSE(hcsparseDestroyDnVec(vec_y));
    CHECK_MCSPARSE(hcsparseDestroySpVec(vec_x));
    CHECK_STATUS(ret);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::sparse_scatter::metax
