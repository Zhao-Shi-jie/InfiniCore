from infinicore.lib import _infinicore
from infinicore.spvec import SpVec
from infinicore.tensor import Tensor


def sparse_scatter(input: SpVec, *, out=None):
    if not isinstance(input, SpVec):
        raise TypeError("sparse_scatter expects a COO SpVec input")

    if out is None:
        return Tensor(_infinicore.sparse_scatter(input._underlying))

    _infinicore.sparse_scatter_(out._underlying, input._underlying)
    return out
