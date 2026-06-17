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
from framework.utils.tensor_utils import infinicore_tensor_from_torch
from sparse_mtx import ValuesFromListSpec, load_csr


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


def _generate_spmm_cases():
    cases = []
    # (rows, cols, n, density)
    configs = [
        # (4096, 4096, 4096, 0.01),  # 2K scale
        (5120, 5120, 5120, 0.01),  # 5K scale
        # (4096, 4096, 4096, 0.02)
    ]
    for rows, cols, n, density in configs:
        crow, col, values = load_csr("spmm", rows, cols, density=density)
        cases.append((rows, cols, n, density, crow, col, values))
    return cases


_TEST_CASES_DATA = _generate_spmm_cases()

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-2, "rtol": 1e-2},
}

# Sparse CSR tensor support is in beta state, so we only test float32 for now.
_TENSOR_DTYPES = [
    infinicore.float32
]


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


def parse_test_cases():
    test_cases = []
    for rows, cols, n, density, crow, col, values in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
            values_spec = ValuesFromListSpec(values, dtype=dtype, name="values")
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
            #             TensorSpec.from_tensor((cols, n), dtype=dtype, name="b"),
            #         ],
            #         kwargs={
            #             "rows": rows,
            #             "cols": cols,
            #             "crow": crow,
            #             "col": col,
            #         },
            #         tolerance=_TOLERANCE_MAP[dtype],
            #         description="SpMM - OUT_OF_PLACE",
            #     )
            # )
            values_spec = ValuesFromListSpec(values, dtype=dtype, name="values")
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
                        TensorSpec.from_tensor((cols, n), dtype=dtype, name="b"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "density": density,
                        "crow": crow,
                        "col": col,
                        "out": TensorSpec.from_tensor(
                            (rows, n), dtype=dtype, name="out"
                        ),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpMM - OUT(out)",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpMM")

    def get_test_cases(self):
        return parse_test_cases()

    # def torch_operator(self, values, sparse, b, *, rows, cols, crow, col, out=None):
    #     del sparse
    #     sparse = csr_to_dense(values, rows, cols, crow, col)
    #     result = torch.matmul(sparse, b)
    #     if out is not None:
    #         out.copy_(result)
    #         return out
    #     return result

    def infinicore_operator(
        self, _values, sparse, b, *, out=None, **_unused
    ):
        return infinicore.spmm(sparse, b, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
