#ifndef __RMS_NORM_KUNLUN_H__
#define __RMS_NORM_KUNLUN_H__

#include "../rms_norm.h"

DESCRIPTOR(kunlun)

#define INSTANTIATE_RMSNORM_KERNEL(BLOCK_SIZE, Tcompute, Tdata, Tweight)          \
    template __global__ void rmsnormKernel<BLOCK_SIZE, Tcompute, Tdata, Tweight>( \
        Tdata * y,                                                                \
        int32_t stride_y,                                                         \
        const Tdata *x,                                                           \
        int32_t stride_x,                                                         \
        const Tweight *w,                                                         \
        uint32_t dim,                                                             \
        float epsilon);

#endif
