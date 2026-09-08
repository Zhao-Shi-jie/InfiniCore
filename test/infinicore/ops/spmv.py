import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from sparse_utils import infinicore_list_on_device, random_csr_indices

_SPARSE_FORMATS = ["csr", "coo"]


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
        col_tensor = infinicore_list_on_device(
            self.col, dtype=infinicore.int64, device=infini_device
        )
        if self.sparse_format == "csr":
            crow_tensor = infinicore_list_on_device(
                self.crow, dtype=infinicore.int64, device=infini_device
            )
            return infinicore.csr_spmat(
                crow_tensor, col_tensor, infini_values, (self.rows, self.cols)
            )

        if self.sparse_format == "coo":
            row_indices = torch.tensor(
                csr_to_coo_rows(self.rows, self.crow),
                dtype=torch.int64,
                device=values.device,
            )
            return infinicore.coo_spmat(
                infinicore.from_torch(row_indices),
                col_tensor,
                infini_values,
                (self.rows, self.cols),
            )

        raise ValueError(f"Unsupported sparse format: {self.sparse_format}")

    def __str__(self):
        return f"{self.name}: spmat(format={self.sparse_format}, rows={self.rows}, cols={self.cols})"


def _generate_spmv_cases():
    cases = []
    # (rows, cols, density, alpha, beta)
    configs = [
        (128, 128, 0.02, 0.5, 1.0),  # Baseline
        (1024, 1024, 0.01, 0.5, 1.0),  # 1K scale
        (1024, 1024, 0.02, 1.0, 0.0),  # 1K scale
        (4096, 4096, 0.01, 0.5, 1.0),  # 4K scale
    ]
    for rows, cols, density, alpha, beta in configs:
        crow, col = random_csr_indices(rows, cols, density, seed=42)
        cases.append((rows, cols, density, crow, col, alpha, beta))
    return cases


_TEST_CASES_DATA = _generate_spmv_cases()

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 5e-3, "rtol": 5e-3},
}

_TENSOR_DTYPES = [
    infinicore.float32,
]


def _use_dense_reference(device):
    return device.type == "mlu"


def csr_to_coo_rows(rows, crow):
    row_indices = []
    for row in range(rows):
        row_indices.extend([row] * (crow[row + 1] - crow[row]))
    return row_indices


def spmv_sparse_reference(values, x, *, rows, cols, crow, col):
    sparse = torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=torch.int64, device=values.device),
        torch.tensor(col, dtype=torch.int64, device=values.device),
        values,
        size=(rows, cols),
    )
    return torch.matmul(sparse, x)


def spmv_dense_reference(values, x, *, rows, cols, crow, col):
    dense = torch.zeros((rows, cols), dtype=values.dtype, device=values.device)
    row_counts = torch.tensor(
        [crow[i + 1] - crow[i] for i in range(rows)],
        dtype=torch.int64,
        device=values.device,
    )
    row_indices = torch.repeat_interleave(
        torch.arange(rows, dtype=torch.int64, device=values.device), row_counts
    )
    col_indices = torch.tensor(col, dtype=torch.int64, device=values.device)
    dense.index_put_((row_indices, col_indices), values, accumulate=True)
    return torch.matmul(dense, x)


def parse_test_cases():
    test_cases = []
    for rows, cols, density, crow, col, alpha, beta in _TEST_CASES_DATA:
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
                            TensorSpec.from_tensor((cols,), dtype=dtype, name="x"),
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
                                (rows,), dtype=dtype, name="out"
                            ),
                        },
                        comparison_target="out",
                        tolerance=_TOLERANCE_MAP[dtype],
                        description=f"SpMV {sparse_format} - OUT(out)",
                    )
                )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpMV")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        values,
        sparse,
        x,
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
        if _use_dense_reference(values.device):
            result = spmv_dense_reference(
                values, x, rows=rows, cols=cols, crow=crow, col=col
            )
        else:
            result = spmv_sparse_reference(
                values, x, rows=rows, cols=cols, crow=crow, col=col
            )
        result = alpha * result
        if out is not None:
            result = result + beta * out
            out.copy_(result)
            return out
        return result

    def infinicore_operator(
        self, _values, sparse, x, *, alpha, beta, out=None, **_unused
    ):
        return infinicore.spmv(sparse, x, alpha=alpha, beta=beta, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
