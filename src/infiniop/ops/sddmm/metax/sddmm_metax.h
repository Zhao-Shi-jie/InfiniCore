#ifndef __SDDMM_METAX_H__
#define __SDDMM_METAX_H__

#include "../sddmm.h"

SDDMM_DESCRIPTOR(metax);

namespace op::sddmm::metax {

infiniStatus_t launchSDDMM(
    const SDDMMInfo &info,
    infiniopSpMatDescriptor_t c_desc,
    void *c_values,
    const void *a,
    const void *b,
    float alpha,
    float beta,
    infiniDtype_t index_dtype,
    void *stream);

} // namespace op::sddmm::metax

#endif // __SDDMM_METAX_H__
