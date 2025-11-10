#include "infinirt_metax.h"
#include "../../utils.h"
#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>

#define CHECK_MACART(RT_API) CHECK_INTERNAL(RT_API, mcSuccess)

namespace infinirt::metax {
infiniStatus_t getDeviceCount(int *count) {
    CHECK_MACART(mcGetDeviceCount(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_MACART(mcSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    CHECK_MACART(mcDeviceSynchronize());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    mcStream_t stream;
    CHECK_MACART(mcStreamCreate(&stream));
    *stream_ptr = stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    CHECK_MACART(mcStreamDestroy((mcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_MACART(mcStreamSynchronize((mcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    CHECK_MACART(mcStreamWaitEvent((mcStream_t)stream, (mcEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    mcEvent_t event;
    CHECK_MACART(mcEventCreate(&event));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_MACART(mcEventRecord((mcEvent_t)event, (mcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    CHECK_MACART(hcEventQuery((hcEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_MACART(hcEventSynchronize((hcEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    CHECK_MACART(hcEventDestroy((hcEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_MACART(hcMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_MACART(hcMallocHost(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    CHECK_MACART(hcFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_MACART(hcFreeHost(ptr));
    return INFINI_STATUS_SUCCESS;
}

hcMemcpyKind toMacaMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        return hcMemcpyHostToDevice;
    case INFINIRT_MEMCPY_D2H:
        return hcMemcpyDeviceToHost;
    case INFINIRT_MEMCPY_D2D:
        return hcMemcpyDeviceToDevice;
    case INFINIRT_MEMCPY_H2H:
        return hcMemcpyHostToHost;
    default:
        return hcMemcpyDefault;
    }
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    CHECK_MACART(hcMemcpy(dst, src, size, toMacaMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_MACART(hcMemcpyAsync(dst, src, size, toMacaMemcpyKind(kind), (mcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    CHECK_MACART(hcMallocAsync(p_ptr, size, (mcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    CHECK_MACART(hcFreeAsync(ptr, (mcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}
} // namespace infinirt::metax
