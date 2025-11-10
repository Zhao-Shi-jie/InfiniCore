#include "../../../utils.h"
#include "../pool.h"
#include "metax_handle.h"
#include <mcblas/mcblas.h>
#include <mcdnn/mcdnn.h>
#include <memory>

#define CHECK_MCBLAS(API) CHECK_INTERNAL(API, MCBLAS_STATUS_SUCCESS)
#define CHECK_MCDNN(API) CHECK_INTERNAL(API, MCDNN_STATUS_SUCCESS)

namespace device::metax {

class Handle::Internal {
    Pool<mcblasHandle_t> mcblas_handles;
    Pool<mcdnnHandle_t> mcdnn_handles;

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

    int _warp_size,
        _max_threads_per_block,
        _block_size[3],
        _grid_size[3];

public:
    Internal(int);
    infiniStatus_t useMcblas(mcStream_t stream, const Fn<mcblasHandle_t> &f) const;
    infiniStatus_t useMcdnn(mcStream_t stream, const Fn<mcdnnHandle_t> &f) const;

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
    int gridSizeX() const;
    int gridSizeY() const;
    int gridSizeZ() const;
};

mcdnnDataType_t getMcdnnDtype(infiniDtype_t dt);

} // namespace device::metax
