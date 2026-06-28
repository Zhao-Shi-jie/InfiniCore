#ifndef __AXPBY_INFO_H__
#define __AXPBY_INFO_H__

#include "../../../utils.h"
#include "../../spvec.h"
#include "../../tensor.h"

namespace op::axpby {

class AxpbyInfo {
    AxpbyInfo() = default;

public:
    size_t n;
    size_t nnz;
    ptrdiff_t incy;
    infiniDtype_t dtype;

    static utils::Result<AxpbyInfo> create(
        infiniopSpVecDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc) {
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->size() == y_desc->numel(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->valuesDesc()->dtype() == y_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(y_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(y_desc->stride(0) != 0, INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<AxpbyInfo>(AxpbyInfo{
            x_desc->size(),
            x_desc->nnz(),
            y_desc->stride(0),
            y_desc->dtype()});
    }
};

} // namespace op::axpby

#endif // __AXPBY_INFO_H__
