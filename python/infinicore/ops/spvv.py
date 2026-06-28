from infinicore.lib import _infinicore
from infinicore.spvec import SpVec
from infinicore.tensor import Tensor


def spvv(a, x, *, out=None):
    if not isinstance(a, SpVec):
        raise TypeError("spvv expects a COO SpVec as the left-hand side")

    if out is None:
        return Tensor(_infinicore.spvv(a._underlying, x._underlying))

    _infinicore.spvv_(
        out._underlying,
        a._underlying,
        x._underlying,
    )
    return out
