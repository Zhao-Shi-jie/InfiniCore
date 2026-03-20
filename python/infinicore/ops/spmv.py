from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def spmv(
    input_x: Tensor,
    values: Tensor,
    row_ptr: Tensor,
    col_indices: Tensor,
    rows: int,
    cols: int,
    nnzs: int,
    *,
    out=None,
) -> Tensor:
    if out is None:
        return Tensor(
            _infinicore.spmv(input_x._underlying, values._underlying, row_ptr._underlying, col_indices._underlying, rows, cols, nnzs)
        )

    _infinicore.spmv_(
        out._underlying, input_x._underlying, values._underlying, row_ptr._underlying, col_indices._underlying, rows, cols, nnzs
    )
    return