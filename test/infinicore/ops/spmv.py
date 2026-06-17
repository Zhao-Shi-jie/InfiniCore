import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from framework.utils.tensor_utils import infinicore_tensor_from_torch
from sparse_mtx import maybe_write_csr, random_csr_indices


class SparseTestCase(TestCase):
    def __str__(self):
        return (
            f"TestCase({self.description} - rows={self.kwargs['rows']}; "
            f"cols={self.kwargs['cols']}; density={self.kwargs['density']:.6f})"
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
        mtx_name=None,
        density=None,
        name="sparse",
    ):
        super().__init__(shape=(rows, cols), dtype=values_spec.dtype, name=name)
        self.values_spec = values_spec
        self.rows = rows
        self.cols = cols
        self.crow = crow
        self.col = col
        self.mtx_name = mtx_name
        self.density = density
        self._cached_values = {}

    def create_torch_tensor(self, device):
        if device not in self._cached_values:
            self._cached_values[device] = self.values_spec.create_torch_tensor(
                device
            ).clone()
        values = self._cached_values[device]
        if self.mtx_name is not None:
            maybe_write_csr(
                self.mtx_name,
                self.rows,
                self.cols,
                self.crow,
                self.col,
                values=values,
                density=self.density,
            )
        infini_values = infinicore_tensor_from_torch(values)
        infini_device = infini_values.device
        crow_tensor = infinicore.from_list(
            self.crow, dtype=infinicore.int64, device=infini_device
        )
        col_tensor = infinicore.from_list(
            self.col, dtype=infinicore.int64, device=infini_device
        )
        return infinicore.csr_spmat(
            crow_tensor, col_tensor, infini_values, (self.rows, self.cols)
        )

    def __str__(self):
        return f"{self.name}: spmat(rows={self.rows}, cols={self.cols})"


def _generate_spmv_cases():
    cases = []
    # (rows, cols, density)
    configs = [
        (128, 128, 0.02),  # Baseline
        (1024, 1024, 0.01),  # 1K scale
        (1024, 1024, 0.02),  # 1K scale
        (4096, 4096, 0.01),  # 4K scale
    ]
    for rows, cols, density in configs:
        crow, col = random_csr_indices(rows, cols, density, seed=42)
        cases.append((rows, cols, density, crow, col))
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
    for rows, cols, density, crow, col in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
            values_spec = CachedTensorSpec.from_tensor(
                (nnz,), dtype=dtype, name="values"
            )
            # test_cases.append(
            #     SparseTestCase(
            #         inputs=[
            #             values_spec,
            #             CsrSpMatSpec(
            #                 values_spec=values_spec,
            #                 rows=rows,
            #                 cols=cols,
            #                 crow=crow,
            #                 col=col,
            #             ),
            #             TensorSpec.from_tensor((cols,), dtype=dtype, name="x"),
            #         ],
            #         kwargs={
            #             "rows": rows,
            #             "cols": cols,
            #             "crow": crow,
            #             "col": col,
            #         },
            #         tolerance=_TOLERANCE_MAP[dtype],
            #         description="SpMV - OUT_OF_PLACE",
            #     )
            # )
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
                            mtx_name="spmv",
                            density=density,
                        ),
                        TensorSpec.from_tensor((cols,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "density": density,
                        "crow": crow,
                        "col": col,
                        "out": TensorSpec.from_tensor((rows,), dtype=dtype, name="out"),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpMV - OUT(out)",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpMV")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self, values, sparse, x, *, rows, cols, density, crow, col, out=None
    ):
        del sparse
        del density
        if _use_dense_reference(values.device):
            result = spmv_dense_reference(
                values, x, rows=rows, cols=cols, crow=crow, col=col
            )
        else:
            result = spmv_sparse_reference(
                values, x, rows=rows, cols=cols, crow=crow, col=col
            )
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(
        self, _values, sparse, x, *, out=None, **_unused
    ):
        return infinicore.spmv(sparse, x, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
