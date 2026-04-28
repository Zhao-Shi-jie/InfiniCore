from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def spmm(a, b, *, alpha=1.0):
    """Sparse Matrix × Dense Matrix multiplication (SpMM).

    Computes ``C = alpha * A * B`` where *A* is a CSR sparse matrix and *B*
    is a dense matrix.

    Parameters
    ----------
    a : SpMat
        CSR sparse matrix of shape ``(m, k)``.
    b : Tensor
        Dense input matrix of shape ``(k, n)``.
    alpha : float, optional
        Scalar multiplier (default ``1.0``).

    Returns
    -------
    Tensor
        Dense output matrix of shape ``(m, n)``.
    """
    return Tensor(_infinicore.spmm(a._underlying, b._underlying, alpha))


def spmm_(c, a, b, *, alpha=1.0, beta=0.0):
    """In-place Sparse Matrix × Dense Matrix multiplication.

    Computes ``C = alpha * A * B + beta * C`` in-place.

    Parameters
    ----------
    c : Tensor
        Dense output matrix of shape ``(m, n)`` — modified in-place.
    a : SpMat
        CSR sparse matrix of shape ``(m, k)``.
    b : Tensor
        Dense input matrix of shape ``(k, n)``.
    alpha : float, optional
        Scalar multiplier for ``A * B`` (default ``1.0``).
    beta : float, optional
        Scalar multiplier for the existing *C* (default ``0.0``).

    Returns
    -------
    Tensor
        The same tensor *c* (for chaining).
    """
    _infinicore.spmm_(c._underlying, a._underlying, b._underlying, alpha, beta)
    return c
