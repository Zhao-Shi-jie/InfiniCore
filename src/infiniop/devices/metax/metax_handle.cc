#include "metax_common.h"

namespace device::metax {
Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>(device_id)) {}

Handle::Handle(int device_id) : Handle(INFINI_DEVICE_METAX, device_id) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

Handle::Internal::Internal(int device_id) {
    hcDeviceProp_t prop;
    hcGetDeviceProperties(&prop, device_id);
    _warp_size = prop.warpSize;
    _max_threads_per_block = prop.maxThreadsPerBlock;
    _block_size[0] = prop.maxThreadsDim[0];
    _block_size[1] = prop.maxThreadsDim[1];
    _block_size[2] = prop.maxThreadsDim[2];
    _grid_size[0] = prop.maxGridSize[0];
    _grid_size[1] = prop.maxGridSize[1];
    _grid_size[2] = prop.maxGridSize[2];
}

infiniStatus_t Handle::Internal::useMcblas(mcStream_t stream, const Fn<mcblasHandle_t> &f) const {
    auto handle = mcblas_handles.pop();
    if (!handle) {
        CHECK_MCBLAS(mcblasCreate(&(*handle)));
    }
    CHECK_MCBLAS(mcblasSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    mcblas_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::Internal::useMcdnn(mcStream_t stream, const Fn<mcdnnHandle_t> &f) const {
    auto handle = mcdnn_handles.pop();
    if (!handle) {
        CHECK_MCDNN(mcdnnCreate(&(*handle)));
    }
    CHECK_MCDNN(mcdnnSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    mcdnn_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

int Handle::Internal::warpSize() const { return _warp_size; }
int Handle::Internal::maxThreadsPerBlock() const { return _max_threads_per_block; }
int Handle::Internal::blockSizeX() const { return _block_size[0]; }
int Handle::Internal::blockSizeY() const { return _block_size[1]; }
int Handle::Internal::blockSizeZ() const { return _block_size[2]; }
int Handle::Internal::gridSizeX() const { return _grid_size[0]; }
int Handle::Internal::gridSizeY() const { return _grid_size[1]; }
int Handle::Internal::gridSizeZ() const { return _grid_size[2]; }

mcdnnDataType_t getMcdnnDtype(infiniDtype_t dt) {
    switch (dt) {
    case INFINI_DTYPE_F16:
        return MCDNN_DATA_HALF;
    case INFINI_DTYPE_F32:
        return MCDNN_DATA_FLOAT;
    case INFINI_DTYPE_F64:
        return MCDNN_DATA_DOUBLE;
    case INFINI_DTYPE_BF16:
        return MCDNN_DATA_BFLOAT16;
    case INFINI_DTYPE_I8:
        return MCDNN_DATA_INT8;
    case INFINI_DTYPE_I32:
        return MCDNN_DATA_INT32;
    case INFINI_DTYPE_I64:
        return MCDNN_DATA_INT64;
    case INFINI_DTYPE_U8:
        return MCDNN_DATA_UINT8;
    default:
        return MCDNN_DATA_FLOAT;
    }
}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(INFINI_DEVICE_METAX, device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::metax
