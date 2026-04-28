from infinicore.lib import _infinicore
from infinicore.spmat import SpMat
from infinicore.tensor import Tensor


def gemm(a, b, *, alpha=1.0, beta=0.0, out=None):
    if isinstance(a, SpMat):
        if out is None:
            return Tensor(
                _infinicore.gemm(a._underlying, b._underlying, alpha, beta)
            )

        _infinicore.gemm_(
            out._underlying,
            a._underlying,
            b._underlying,
            alpha,
            beta,
        )
        return out

    raise TypeError("gemm currently expects a CSR SpMat as the left-hand side")
