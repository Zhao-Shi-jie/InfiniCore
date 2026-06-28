from infinicore.lib import _infinicore
from infinicore.spmat import SpMat
from infinicore.tensor import Tensor


def spmv(a, x, *, alpha=1.0, beta=0.0, out=None):
    if not isinstance(a, SpMat):
        raise TypeError("spmv expects a CSR SpMat as the left-hand side")

    if out is None:
        return Tensor(_infinicore.spmv(a._underlying, x._underlying, alpha, beta))

    _infinicore.spmv_(
        out._underlying,
        a._underlying,
        x._underlying,
        alpha,
        beta,
    )
    return out
