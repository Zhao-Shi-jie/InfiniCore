import infinicore.device
import infinicore.dtype
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


class SpVec:
    _underlying: _infinicore.SpVec

    def __init__(self, underlying):
        self._underlying = underlying

    @property
    def size(self):
        return self._underlying.size

    @property
    def nnz(self):
        return self._underlying.nnz

    @property
    def shape(self):
        return [self.size]

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
    def indices(self):
        return Tensor(self._underlying.indices)

    @property
    def values(self):
        return Tensor(self._underlying.values)


def coo_spvec(indices, values, size):
    return SpVec(_infinicore.coo_spvec(indices._underlying, values._underlying, size))
