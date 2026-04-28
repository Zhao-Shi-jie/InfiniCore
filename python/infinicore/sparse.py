"""
SpMat — sparse matrix in CSR (Compressed Sparse Row) format.

This module provides a thin Python wrapper around the C++ ``SpMat`` class,
mirroring the cuSPARSE ``cusparseSpMatDescr_t`` descriptor pattern while
reusing the existing ``infinicore.Tensor`` for all underlying storage.

Typical usage::

    import numpy as np
    import infinicore
    from infinicore.sparse import SpMat

    # Build a simple 4×4 CSR matrix from dense data
    dense = np.array([[1, 0, 0, 2],
                      [0, 3, 0, 0],
                      [0, 0, 4, 0],
                      [5, 0, 6, 0]], dtype=np.float32)
    sp = SpMat.from_dense(dense, device=infinicore.device("cpu", 0))

    # Multiply with a dense tensor
    B = infinicore.from_numpy(np.eye(4, dtype=np.float32))
    C = infinicore.spmm(sp, B)  # same as dense @ B
"""

from __future__ import annotations

import numpy as np

import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


class SpMat:
    """Sparse matrix in CSR (Compressed Sparse Row) format.

    Internally holds three ``infinicore.Tensor`` objects:

    * ``row_offsets`` — shape ``[rows + 1]``, dtype ``int32``
    * ``col_indices`` — shape ``[nnz]``,     dtype ``int32``
    * ``values``      — shape ``[nnz]``,     dtype *T*

    Parameters
    ----------
    row_offsets : Tensor
        1-D tensor of shape ``[rows + 1]``, dtype ``int32``.
    col_indices : Tensor
        1-D tensor of shape ``[nnz]``, dtype ``int32``.
    values : Tensor
        1-D tensor of shape ``[nnz]`` containing non-zero values.
    rows : int
        Number of rows in the sparse matrix.
    cols : int
        Number of columns in the sparse matrix.
    """

    def __init__(
        self,
        row_offsets: Tensor,
        col_indices: Tensor,
        values: Tensor,
        rows: int,
        cols: int,
    ) -> None:
        self._underlying = _infinicore.SpMat(
            row_offsets._underlying,
            col_indices._underlying,
            values._underlying,
            rows,
            cols,
        )
        # Keep Python-level references to avoid premature GC
        self._row_offsets = row_offsets
        self._col_indices = col_indices
        self._values = values

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def row_offsets(self) -> Tensor:
        """Row-pointer tensor ``[rows + 1]``, dtype int32."""
        return self._row_offsets

    @property
    def col_indices(self) -> Tensor:
        """Column-index tensor ``[nnz]``, dtype int32."""
        return self._col_indices

    @property
    def values(self) -> Tensor:
        """Non-zero value tensor ``[nnz]``."""
        return self._values

    @property
    def rows(self) -> int:
        return self._underlying.rows

    @property
    def cols(self) -> int:
        return self._underlying.cols

    @property
    def nnz(self) -> int:
        return self._underlying.nnz

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def device(self):
        return infinicore.device._from_infinicore_device(self._underlying.device)

    @property
    def dtype(self):
        return infinicore.dtype(self._underlying.dtype)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dense(
        cls,
        dense: np.ndarray,
        *,
        device: infinicore.device = None,
        dtype: infinicore.dtype = None,
    ) -> "SpMat":
        """Convert a 2-D NumPy array to a CSR SpMat.

        Parameters
        ----------
        dense : np.ndarray
            2-D array of shape ``(m, k)``.
        device : infinicore.device, optional
            Target device (default: CPU).
        dtype : infinicore.dtype, optional
            Value dtype (default: inferred from *dense*).
        """
        if dense.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {dense.ndim}-D")

        if device is None:
            device = infinicore.device("cpu", 0)
        if dtype is None:
            dtype = infinicore.float32

        # Convert to float32 if needed so scipy / numpy can process it
        np_dtype = np.float32 if dtype == infinicore.float32 else np.float64
        dense = np.asarray(dense, dtype=np_dtype)

        m, k = dense.shape
        mask = dense != 0
        rows_np, cols_np = np.where(mask)
        vals_np = dense[mask].astype(np_dtype)
        nnz = len(vals_np)

        # Build row_offsets
        row_counts = np.bincount(rows_np, minlength=m).astype(np.int32)
        row_offsets_np = np.zeros(m + 1, dtype=np.int32)
        np.cumsum(row_counts, out=row_offsets_np[1:])

        col_indices_np = cols_np.astype(np.int32)

        ro_t = infinicore.from_numpy(
            row_offsets_np, dtype=infinicore.int32, device=device
        )
        ci_t = infinicore.from_numpy(
            col_indices_np, dtype=infinicore.int32, device=device
        )
        v_t = infinicore.from_numpy(vals_np, dtype=dtype, device=device)

        return cls(ro_t, ci_t, v_t, m, k)

    @classmethod
    def from_csr(
        cls,
        row_offsets: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
        rows: int,
        cols: int,
        *,
        device: infinicore.device = None,
        dtype: infinicore.dtype = None,
    ) -> "SpMat":
        """Construct a SpMat from raw CSR arrays.

        Parameters
        ----------
        row_offsets : np.ndarray
            1-D int32 array of length ``rows + 1``.
        col_indices : np.ndarray
            1-D int32 array of length ``nnz``.
        values : np.ndarray
            1-D float array of length ``nnz``.
        rows, cols : int
            Matrix dimensions.
        device : infinicore.device, optional
        dtype : infinicore.dtype, optional
        """
        if device is None:
            device = infinicore.device("cpu", 0)
        if dtype is None:
            dtype = infinicore.float32

        ro_t = infinicore.from_numpy(
            np.asarray(row_offsets, dtype=np.int32),
            dtype=infinicore.int32,
            device=device,
        )
        ci_t = infinicore.from_numpy(
            np.asarray(col_indices, dtype=np.int32),
            dtype=infinicore.int32,
            device=device,
        )
        v_t = infinicore.from_numpy(
            np.asarray(values, dtype=np.float32 if dtype == infinicore.float32 else np.float64),
            dtype=dtype,
            device=device,
        )
        return cls(ro_t, ci_t, v_t, rows, cols)

    def __repr__(self) -> str:
        return (
            f"SpMat(shape=({self.rows}, {self.cols}), nnz={self.nnz}, "
            f"dtype={self.dtype}, device={self.device}, format=CSR)"
        )
