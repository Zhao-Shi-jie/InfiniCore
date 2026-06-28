from infinicore.lib import _infinicore
from infinicore.spvec import SpVec
from infinicore.tensor import Tensor


def axpby(x, y: Tensor, *, alpha=1.0, beta=1.0, out=None):
    if not isinstance(x, SpVec):
        raise TypeError("axpby expects a COO SpVec as the sparse input")
    if out is None:
        return Tensor(_infinicore.axpby(x._underlying, y._underlying, alpha, beta))

    out.copy_(y)
    _infinicore.axpby_(x._underlying, out._underlying, alpha, beta)
    return out
