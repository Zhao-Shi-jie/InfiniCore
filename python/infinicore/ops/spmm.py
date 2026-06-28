from infinicore.lib import _infinicore
from infinicore.spmat import SpMat
from infinicore.tensor import Tensor


def spmm(a, b, *, alpha=1.0, beta=0.0, out=None):
    if not isinstance(a, SpMat):
        raise TypeError("spmm expects a CSR SpMat as the left-hand side")

    if out is None:
        return Tensor(_infinicore.spmm(a._underlying, b._underlying, alpha, beta))

    _infinicore.spmm_(
        out._underlying,
        a._underlying,
        b._underlying,
        alpha,
        beta,
    )
    return out
