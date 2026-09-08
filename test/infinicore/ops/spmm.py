import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)
from sparse_utils import infinicore_list_on_device, random_csr_indices

_SPARSE_FORMATS = ["csr", "ell", "sell", "sell_sigma_c"]
_SELL_SLICE_HEIGHT = 16
_SELL_SIGMA = 64


class SparseTestCase(TestCase):
    def __str__(self):
        return (
            f"TestCase({self.description} - rows={self.kwargs['rows']}; "
            f"cols={self.kwargs['cols']}; density={self.kwargs['density']:.6f}; "
            f"format={self.kwargs['sparse_format']}; "
            f"alpha={self.kwargs['alpha']}; beta={self.kwargs['beta']})"
        )


class CachedTensorSpec(TensorSpec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    @classmethod
    def from_tensor(cls, shape, strides=None, dtype=None, init_mode=None, **kwargs):
        if init_mode is None:
            return cls(shape=shape, dtype=dtype, strides=strides, **kwargs)
        return cls(
            shape=shape, dtype=dtype, strides=strides, init_mode=init_mode, **kwargs
        )

    def create_torch_tensor(self, device):
        if device not in self._cache:
            self._cache[device] = super().create_torch_tensor(device)
        return self._cache[device]


class CsrSpMatSpec(TensorSpec):
    def __init__(
        self,
        *,
        values_spec,
        rows,
        cols,
        crow,
        col,
        sparse_format="csr",
        name="sparse",
    ):
        super().__init__(shape=(rows, cols), dtype=values_spec.dtype, name=name)
        self.values_spec = values_spec
        self.rows = rows
        self.cols = cols
        self.crow = crow
        self.col = col
        self.sparse_format = sparse_format
        self._cached_values = {}

    def create_torch_tensor(self, device):
        if device not in self._cached_values:
            self._cached_values[device] = self.values_spec.create_torch_tensor(
                device
            ).clone()
        values = self._cached_values[device]
        infini_values = infinicore.from_torch(values)
        infini_device = infini_values.device
        if self.sparse_format == "csr":
            crow_tensor = infinicore_list_on_device(
                self.crow, dtype=infinicore.int64, device=infini_device
            )
            col_tensor = infinicore_list_on_device(
                self.col, dtype=infinicore.int64, device=infini_device
            )
            return infinicore.csr_spmat(
                crow_tensor, col_tensor, infini_values, (self.rows, self.cols)
            )

        if self.sparse_format == "ell":
            ell_values, ell_col, ell_width = csr_to_ell(
                values, self.rows, self.crow, self.col
            )
            return infinicore.ell_spmat(
                infinicore.from_torch(ell_col),
                infinicore.from_torch(ell_values),
                (self.rows, self.cols),
                ell_width,
                len(self.col),
            )

        if self.sparse_format == "sell":
            sell_values, slice_offsets, sell_col, _, _ = csr_to_sell(
                values, self.rows, self.crow, self.col, _SELL_SLICE_HEIGHT
            )
            return infinicore.sell_spmat(
                infinicore.from_torch(slice_offsets),
                infinicore.from_torch(sell_col),
                infinicore.from_torch(sell_values),
                (self.rows, self.cols),
                _SELL_SLICE_HEIGHT,
                len(self.col),
            )

        if self.sparse_format == "sell_sigma_c":
            row_order = sigma_sorted_rows(self.rows, self.crow, _SELL_SIGMA)
            sell_values, slice_offsets, sell_col, row_indices, _ = csr_to_sell(
                values,
                self.rows,
                self.crow,
                self.col,
                _SELL_SLICE_HEIGHT,
                row_order,
            )
            return infinicore.sell_sigma_c_spmat(
                infinicore.from_torch(slice_offsets),
                infinicore.from_torch(sell_col),
                infinicore.from_torch(row_indices),
                infinicore.from_torch(sell_values),
                (self.rows, self.cols),
                _SELL_SLICE_HEIGHT,
                _SELL_SIGMA,
                len(self.col),
            )

        raise ValueError(f"Unsupported sparse format: {self.sparse_format}")

    def __str__(self):
        return f"{self.name}: spmat(format={self.sparse_format}, rows={self.rows}, cols={self.cols})"


def _generate_spmm_cases():
    cases = []
    # (rows, cols, n, density, alpha, beta)
    configs = [
        (128, 128, 128, 0.01, 0.5, 1.0),  # Baseline small test
        (1024, 1024, 1024, 0.01, 0.5, 1.0),  # 1K scale
        (1024, 1024, 1024, 0.02, 0.53, 1.05),  # 1K scale with higher density
        (4096, 2048, 4096, 0.01, 0.5, 1.0),  # 2K scale
    ]
    for rows, cols, n, density, alpha, beta in configs:
        crow, col = random_csr_indices(rows, cols, density, seed=42)
        cases.append((rows, cols, n, density, crow, col, alpha, beta))
    return cases


_TEST_CASES_DATA = _generate_spmm_cases()

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-2, "rtol": 1e-2},
}

# Sparse CSR tensor support is in beta state, so we only test float32 for now.
_TENSOR_DTYPES = [infinicore.float32]


def csr_to_dense(values, rows, cols, crow, col):
    device = values.device
    crow_tensor = torch.tensor(crow, dtype=torch.int64, device=device)
    col_tensor = torch.tensor(col, dtype=torch.int64, device=device)
    row_counts = crow_tensor[1:] - crow_tensor[:-1]
    row_tensor = torch.repeat_interleave(
        torch.arange(rows, dtype=torch.int64, device=device), row_counts
    )
    dense = torch.zeros((rows, cols), dtype=values.dtype, device=device)
    dense.index_put_((row_tensor, col_tensor), values, accumulate=True)
    return dense


def csr_row_lengths(rows, crow):
    return [crow[row + 1] - crow[row] for row in range(rows)]


def csr_to_ell(values, rows, crow, col):
    row_lengths = csr_row_lengths(rows, crow)
    ell_width = max(row_lengths) if row_lengths else 0
    ell_values = torch.zeros(
        (rows, ell_width), dtype=values.dtype, device=values.device
    )
    ell_col = torch.zeros((rows, ell_width), dtype=torch.int64, device=values.device)
    for row in range(rows):
        for slot, ptr in enumerate(range(crow[row], crow[row + 1])):
            ell_values[row, slot] = values[ptr]
            ell_col[row, slot] = col[ptr]
    return ell_values, ell_col, ell_width


def sigma_sorted_rows(rows, crow, sigma):
    lengths = csr_row_lengths(rows, crow)
    order = []
    for begin in range(0, rows, sigma):
        window = list(range(begin, min(begin + sigma, rows)))
        window.sort(key=lambda row: lengths[row], reverse=True)
        order.extend(window)
    return order


def csr_to_sell(values, rows, crow, col, slice_height, row_order=None):
    if row_order is None:
        row_order = list(range(rows))

    row_lengths = csr_row_lengths(rows, crow)
    num_slices = (rows + slice_height - 1) // slice_height
    slice_offsets = [0]
    sell_values = []
    sell_col = []

    for slice_id in range(num_slices):
        begin = slice_id * slice_height
        storage_rows = row_order[begin : begin + slice_height]
        slice_width = max((row_lengths[row] for row in storage_rows), default=0)
        for slot in range(slice_width):
            for row_in_slice in range(slice_height):
                if row_in_slice >= len(storage_rows):
                    sell_values.append(
                        torch.zeros((), dtype=values.dtype, device=values.device)
                    )
                    sell_col.append(0)
                    continue
                row = storage_rows[row_in_slice]
                row_nnz = row_lengths[row]
                if slot < row_nnz:
                    ptr = crow[row] + slot
                    sell_values.append(values[ptr])
                    sell_col.append(col[ptr])
                else:
                    sell_values.append(
                        torch.zeros((), dtype=values.dtype, device=values.device)
                    )
                    sell_col.append(0)
        slice_offsets.append(len(sell_values))

    if sell_values:
        values_tensor = torch.stack(sell_values).to(dtype=values.dtype)
    else:
        values_tensor = torch.empty((0,), dtype=values.dtype, device=values.device)
    col_tensor = torch.tensor(sell_col, dtype=torch.int64, device=values.device)
    offsets_tensor = torch.tensor(
        slice_offsets, dtype=torch.int64, device=values.device
    )
    row_indices_tensor = torch.tensor(
        row_order, dtype=torch.int64, device=values.device
    )
    return values_tensor, offsets_tensor, col_tensor, row_indices_tensor, num_slices


def parse_test_cases():
    test_cases = []
    for rows, cols, n, density, crow, col, alpha, beta in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
            for sparse_format in _SPARSE_FORMATS:
                values_spec = CachedTensorSpec.from_tensor(
                    (nnz,), dtype=dtype, name="values"
                )
                test_cases.append(
                    SparseTestCase(
                        inputs=[
                            values_spec,
                            CsrSpMatSpec(
                                values_spec=values_spec,
                                rows=rows,
                                cols=cols,
                                crow=crow,
                                col=col,
                                sparse_format=sparse_format,
                            ),
                            TensorSpec.from_tensor((cols, n), dtype=dtype, name="b"),
                        ],
                        kwargs={
                            "rows": rows,
                            "cols": cols,
                            "density": density,
                            "crow": crow,
                            "col": col,
                            "sparse_format": sparse_format,
                            "alpha": alpha,
                            "beta": beta,
                            "out": TensorSpec.from_tensor(
                                (rows, n), dtype=dtype, name="out"
                            ),
                        },
                        comparison_target="out",
                        tolerance=_TOLERANCE_MAP[dtype],
                        description=f"SpMM {sparse_format} - OUT(out)",
                    )
                )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpMM")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        values,
        sparse,
        b,
        *,
        rows,
        cols,
        density,
        crow,
        col,
        sparse_format,
        alpha,
        beta,
        out=None,
    ):
        del sparse
        del density
        del sparse_format
        sparse = csr_to_dense(values, rows, cols, crow, col)
        result = alpha * torch.matmul(sparse, b)
        if out is not None:
            result = result + beta * out
            out.copy_(result)
            return out
        return result

    def infinicore_operator(
        self, _values, sparse, b, *, alpha, beta, out=None, **_unused
    ):
        return infinicore.spmm(sparse, b, alpha=alpha, beta=beta, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
