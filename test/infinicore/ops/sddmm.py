import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    InfiniDeviceEnum,
    TensorSpec,
    TestCase,
)
from framework.utils.tensor_utils import infinicore_tensor_from_torch
from sparse_mtx import maybe_write_csr, random_csr_indices


def _generate_sddmm_cases():
    cases = []
    # (rows, cols, k, density, alpha, beta)
    configs = [
        (128, 128, 128, 0.01, 1.0, 0.0),
        (1024, 1024, 1024, 0.01, 0.5, 1.0),
        (1024, 1024, 1024, 0.02, 1.0, 0.0),
        (4096, 2048, 4096, 0.01, -1.25, 0.25),
    ]
    for rows, cols, k, density, alpha, beta in configs:
        crow, col = random_csr_indices(rows, cols, density, seed=42)
        cases.append((rows, cols, k, density, crow, col, alpha, beta))
    return cases


_TEST_CASES_DATA = _generate_sddmm_cases()

_TENSOR_DTYPES = [infinicore.float32]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-2, "rtol": 1e-2},
}


def sampled_mm(a, b, values, rows, crow, col, alpha, beta, *, force_cpu=False):
    output_device = values.device
    if force_cpu:
        a = a.cpu()
        b = b.cpu()
        values = values.cpu()
    mm = torch.matmul(a, b)
    result = values.clone()
    for row in range(rows):
        for ptr in range(crow[row], crow[row + 1]):
            result[ptr] = alpha * mm[row, col[ptr]] + beta * values[ptr]
    return result.to(output_device)


def _use_dense_reference(device, target_device):
    return device.type == "mlu" or target_device == InfiniDeviceEnum.METAX


def sddmm_sparse_reference(values, a, b, *, rows, cols, crow, col, alpha, beta):
    sparse = torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=torch.int64, device=values.device),
        torch.tensor(col, dtype=torch.int64, device=values.device),
        values,
        size=(rows, cols),
    )
    return torch.sparse.sampled_addmm(sparse, a, b, beta=beta, alpha=alpha).values()


def sddmm_dense_reference(
    values, a, b, *, rows, cols, crow, col, alpha, beta, force_cpu=False
):
    return sampled_mm(a, b, values, rows, crow, col, alpha, beta, force_cpu=force_cpu)


class SparseTestCase(TestCase):
    def __str__(self):
        return (
            f"TestCase({self.description} - "
            f"rows={self.kwargs['rows']}; cols={self.kwargs['cols']}; "
            f"k={self.kwargs['k']}; density={self.kwargs['density']:.6f})"
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


def parse_test_cases():
    test_cases = []
    for rows, cols, k, density, crow, col, alpha, beta in _TEST_CASES_DATA:
        nnz = len(col)
        for dtype in _TENSOR_DTYPES:
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
                            mtx_name="sddmm",
                            density=density,
                        ),
                        TensorSpec.from_tensor((rows, k), dtype=dtype, name="a"),
                        TensorSpec.from_tensor((k, cols), dtype=dtype, name="b"),
                    ],
                    kwargs={
                        "rows": rows,
                        "cols": cols,
                        "k": k,
                        "density": density,
                        "crow": crow,
                        "col": col,
                        "alpha": alpha,
                        "beta": beta,
                    },
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SDDMM - INPLACE",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SDDMM")
        self._target_device = None

    def get_test_cases(self):
        return parse_test_cases()

    def run_test(self, device, test_case, config):
        self._target_device = device
        try:
            return super().run_test(device, test_case, config)
        finally:
            self._target_device = None

    def torch_operator(
        self, values, sparse, a, b, *, rows, cols, k, density, crow, col, alpha, beta
    ):
        del sparse
        del k
        del density
        if _use_dense_reference(values.device, self._target_device):
            return sddmm_dense_reference(
                values,
                a,
                b,
                rows=rows,
                cols=cols,
                crow=crow,
                col=col,
                alpha=alpha,
                beta=beta,
                force_cpu=True,
            )
        return sddmm_sparse_reference(
            values,
            a,
            b,
            rows=rows,
            cols=cols,
            crow=crow,
            col=col,
            alpha=alpha,
            beta=beta,
        )

    def infinicore_operator(
        self, _values, sparse, a, b, *, alpha, beta, **_unused
    ):
        return infinicore.sddmm(sparse, a, b, alpha=alpha, beta=beta).values


if __name__ == "__main__":
    torch.manual_seed(42)
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
