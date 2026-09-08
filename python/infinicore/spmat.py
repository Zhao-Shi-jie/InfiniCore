import infinicore.device
import infinicore.dtype
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


class SpMat:
    _underlying: _infinicore.SpMat

    def __init__(self, underlying, tensors=None):
        self._underlying = underlying
        self._tensors = list(tensors or [])

    @property
    def rows(self):
        return self._underlying.rows

    @property
    def format(self):
        return self._underlying.format

    @property
    def cols(self):
        return self._underlying.cols

    @property
    def nnz(self):
        return self._underlying.nnz

    @property
    def ell_width(self):
        return self._underlying.ell_width

    @property
    def slice_height(self):
        return self._underlying.slice_height

    @property
    def sigma(self):
        return self._underlying.sigma

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
    def slice_offsets(self):
        return Tensor(self._underlying.slice_offsets)

    @property
    def row_indices(self):
        return Tensor(self._underlying.row_indices)

    @property
    def values(self):
        return Tensor(self._underlying.values)

    def __matmul__(self, other):
        if other.ndim == 1:
            from infinicore.ops.spmv import spmv

            return spmv(self, other)

        from infinicore.ops.spmm import spmm

        return spmm(self, other)


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
        ),
        [crow_indices, col_indices, values],
    )


def coo_spmat(row_indices, col_indices, values, size):
    if len(size) != 2:
        raise ValueError("COO sparse matrix size must be a 2-tuple/list")
    return SpMat(
        _infinicore.coo_spmat(
            row_indices._underlying,
            col_indices._underlying,
            values._underlying,
            size[0],
            size[1],
        ),
        [row_indices, col_indices, values],
    )


def ell_spmat(col_indices, values, size, ell_width, nnz):
    if len(size) != 2:
        raise ValueError("ELL sparse matrix size must be a 2-tuple/list")
    return SpMat(
        _infinicore.ell_spmat(
            col_indices._underlying,
            values._underlying,
            size[0],
            size[1],
            ell_width,
            nnz,
        ),
        [col_indices, values],
    )


def sell_spmat(slice_offsets, col_indices, values, size, slice_height, nnz):
    if len(size) != 2:
        raise ValueError("SELL sparse matrix size must be a 2-tuple/list")
    return SpMat(
        _infinicore.sell_spmat(
            slice_offsets._underlying,
            col_indices._underlying,
            values._underlying,
            size[0],
            size[1],
            slice_height,
            nnz,
        ),
        [slice_offsets, col_indices, values],
    )


def sell_sigma_c_spmat(
    slice_offsets, col_indices, row_indices, values, size, slice_height, sigma, nnz
):
    if len(size) != 2:
        raise ValueError("SELL-sigma-c sparse matrix size must be a 2-tuple/list")
    return SpMat(
        _infinicore.sell_sigma_c_spmat(
            slice_offsets._underlying,
            col_indices._underlying,
            row_indices._underlying,
            values._underlying,
            size[0],
            size[1],
            slice_height,
            sigma,
            nnz,
        ),
        [slice_offsets, col_indices, row_indices, values],
    )
