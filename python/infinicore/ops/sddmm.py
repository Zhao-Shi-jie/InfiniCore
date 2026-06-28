from infinicore.lib import _infinicore
from infinicore.spmat import SpMat
from infinicore.tensor import Tensor


def sddmm(c: SpMat, a: Tensor, b: Tensor, *, alpha=1.0, beta=0.0):
    if not isinstance(c, SpMat):
        raise TypeError("sddmm expects a CSR SpMat as the sampled output")

    _infinicore.sddmm_(c._underlying, a._underlying, b._underlying, alpha, beta)
    return c
