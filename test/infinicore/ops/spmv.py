import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from framework.utils.tensor_utils import infinicore_tensor_from_torch


def _summarize_sparse_data(rows, cols, crow, col, limit=6):
    def preview(values):
        if len(values) <= limit:
            return str(values)
        head = ", ".join(str(v) for v in values[: limit // 2])
        tail = ", ".join(str(v) for v in values[-(limit // 2) :])
        return f"[{head}, ..., {tail}]"

    return [
        f"rows={rows}",
        f"cols={cols}",
        f"crow={preview(crow)}",
        f"col={preview(col)}",
    ]


class SparseTestCase(TestCase):
    def __str__(self):
        input_str = "; ".join(str(inp) for inp in self.inputs)
        kwargs_strs = _summarize_sparse_data(
            self.kwargs["rows"],
            self.kwargs["cols"],
            self.kwargs["crow"],
            self.kwargs["col"],
        )
        out = self.kwargs.get("out")
        if out is not None:
            kwargs_strs.append(f"out={out}")
        return (
            f"TestCase({self.description} - inputs=[{input_str}], "
            f"kwargs={{{'; '.join(kwargs_strs)}}})"
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
    def __init__(self, *, values_spec, rows, cols, crow, col, name="sparse"):
        super().__init__(shape=(rows, cols), dtype=values_spec.dtype, name=name)
        self.values_spec = values_spec
        self.rows = rows
        self.cols = cols
        self.crow = crow
        self.col = col
        self._cached_values = {}

    def create_torch_tensor(self, device):
        if device not in self._cached_values:
            self._cached_values[device] = self.values_spec.create_torch_tensor(
                device
            ).clone()
        values = self._cached_values[device]
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
    random.seed(42)
    # (rows, cols, density)
    configs = [
        (5120, 5120, 0.01),  # 5K scale
        (128, 128, 0.02),  # Baseline
        (1024, 1024, 0.02),  # 1K scale
        (4096, 4096, 0.01),  # 4K scale
        (5120, 5120, 0.01),  # 5K scale
    ]
    for rows, cols, density in configs:
        crow = [0]
        col = []
        for _ in range(rows):
            nnz_row = int(cols * density)
            if nnz_row > 0:
                col.extend(sorted(random.sample(range(cols), nnz_row)))
            crow.append(len(col))
        cases.append((rows, cols, crow, col))
    return cases


_TEST_CASES_DATA = _generate_spmv_cases()

# _TEST_CASES_DATA = [
#     (3, 4, [0, 2, 3, 5], [0, 2, 1, 0, 3]),
#     (4, 5, [0, 1, 1, 3, 4], [2, 0, 4, 1]),
# ]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [
    # infinicore.float16,
    # infinicore.bfloat16,
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
    for rows, cols, crow, col in _TEST_CASES_DATA:
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
                        ),
                        TensorSpec.from_tensor((cols,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
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

    def torch_operator(self, values, sparse, x, *, rows, cols, crow, col, out=None):
        del sparse
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
        self, _values, sparse, x, *, rows, cols, crow, col, out=None
    ):
        return infinicore.spmv(sparse, x, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
