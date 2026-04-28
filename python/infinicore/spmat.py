import infinicore.device
import infinicore.dtype
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


class SpMat:
    _underlying: _infinicore.SpMat

    def __init__(self, underlying):
        self._underlying = underlying

    @property
    def rows(self):
        return self._underlying.rows

    @property
    def cols(self):
        return self._underlying.cols

    @property
    def nnz(self):
        return self._underlying.nnz

    @property
    def shape(self):
        return [self.rows, self.cols]

    @property
    def dtype(self):
        return infinicore.dtype.dtype(self._underlying.dtype)

    @property
    def index_dtype(self):
        return infinicore.dtype.dtype(self._underlying.index_dtype)

    @property
    def device(self):
        return infinicore.device._from_infinicore_device(self._underlying.device)

    @property
    def crow_indices(self):
        return Tensor(self._underlying.crow_indices)

    @property
    def col_indices(self):
        return Tensor(self._underlying.col_indices)

    @property
    def values(self):
        return Tensor(self._underlying.values)

    def __matmul__(self, other):
        from infinicore.ops.gemm import gemm

        return gemm(self, other)


def csr_spmat(crow_indices, col_indices, values, size):
    if len(size) != 2:
        raise ValueError("CSR sparse matrix size must be a 2-tuple/list")
    return SpMat(
        _infinicore.csr_spmat(
            crow_indices._underlying,
            col_indices._underlying,
            values._underlying,
            size[0],
            size[1],
        )
    )
