from infinicore.lib import _infinicore
from infinicore.spvec import SpVec
from infinicore.tensor import Tensor


def sparse_gather(pattern: SpVec, input: Tensor, *, out=None):
    if not isinstance(pattern, SpVec):
        raise TypeError("sparse_gather expects a COO SpVec pattern")

    if out is None:
        return Tensor(_infinicore.sparse_gather(pattern._underlying, input._underlying))

    _infinicore.sparse_gather_(
        out._underlying,
        pattern._underlying,
        input._underlying,
    )
    return out
