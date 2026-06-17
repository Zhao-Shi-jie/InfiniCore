import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from framework.utils.tensor_utils import infinicore_tensor_from_torch
from sparse_mtx import maybe_write_spvec, random_spvec_indices


class SparseTestCase(TestCase):
    def __str__(self):
        size = self.kwargs["size"]
        return (
            f"TestCase({self.description} - size={size}; "
            f"density={self.kwargs['density']:.6f})"
        )


def _generate_cases():
    configs = [
        # (128, 0.03, 1.0, 0.0),
        # (1024, 0.02, 1.0, 0.0),
        (1024, 0.01, -1.25, 0.25),
        # (4096, 0.01, 0.5, 1.0),
    ]
    cases = []
    for size, density, alpha, beta in configs:
        indices = random_spvec_indices(size, density, seed=42)
        cases.append((size, density, indices, alpha, beta))
    return cases


_TEST_CASES_DATA = _generate_cases()
_TENSOR_DTYPES = [infinicore.float32]
_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-4},
}


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


class SpVecSpec(TensorSpec):
    def __init__(
        self,
        *,
        values_spec,
        size,
        indices,
        mtx_name=None,
        density=None,
        name="sparse",
    ):
        super().__init__(shape=(size,), dtype=values_spec.dtype, name=name)
        self.values_spec = values_spec
        self.size = size
        self.indices = indices
        self.mtx_name = mtx_name
        self.density = density
        self._cached_values = {}

    def create_torch_tensor(self, device):
        if device not in self._cached_values:
            self._cached_values[device] = self.values_spec.create_torch_tensor(device)
        values = self._cached_values[device]
        if self.mtx_name is not None:
            maybe_write_spvec(
                self.mtx_name,
                self.size,
                self.indices,
                values=values,
                density=self.density,
            )
        infini_values = infinicore_tensor_from_torch(values)
        indices_tensor = infinicore.from_list(
            self.indices, dtype=infinicore.int32, device=infini_values.device
        )
        return infinicore.coo_spvec(indices_tensor, infini_values, self.size)

    def __str__(self):
        return f"{self.name}: spvec(size={self.size})"


def parse_test_cases():
    test_cases = []
    for size, density, indices, alpha, beta in _TEST_CASES_DATA:
        nnz = len(indices)
        for dtype in _TENSOR_DTYPES:
            values_spec = CachedTensorSpec.from_tensor((nnz,), dtype=dtype, name="values")
            test_cases.append(
                SparseTestCase(
                    inputs=[
                        values_spec,
                        SpVecSpec(
                            values_spec=values_spec,
                            size=size,
                            indices=indices,
                            mtx_name="axpby",
                            density=density,
                        ),
                        TensorSpec.from_tensor((size,), dtype=dtype, name="y"),
                    ],
                    kwargs={
                        "size": size,
                        "density": density,
                        "indices": indices,
                        "alpha": alpha,
                        "beta": beta,
                    },
                    comparison_target=2,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="Axpby - INPLACE",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Axpby")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, values, sparse, y, *, size, density, indices, alpha, beta):
        del sparse
        del density
        result = beta * y.clone()
        result[torch.tensor(indices, dtype=torch.int64, device=values.device)] += alpha * values
        y.copy_(result)
        return y

    def infinicore_operator(self, _values, sparse, y, *, alpha, beta, **_unused):
        return infinicore.axpby(sparse, y, alpha=alpha, beta=beta, out=y)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
