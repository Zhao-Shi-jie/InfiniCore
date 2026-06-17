import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from framework.utils.tensor_utils import infinicore_tensor_from_torch
from sparse_mtx import maybe_write_spvec, random_spvec_indices

_TEST_CASES_DATA = [
    # size, density
    # (128, 0.01), 
    # (1024, 0.01),
    (1024, 0.02),
    # (4096, 0.01),
]

_TENSOR_DTYPES = [infinicore.float32]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-4},
}

def _use_dense_reference(device):
    return device.type == "mlu"


def sparse_gather_sparse_reference(x, *, size, indices):
    indices_tensor = torch.tensor(indices, dtype=torch.int64, device=x.device)
    values = torch.ones(len(indices), dtype=x.dtype, device=x.device)
    torch.sparse_coo_tensor(
        indices_tensor.unsqueeze(0),
        values,
        size=(size,),
        device=x.device,
    )
    return x[indices_tensor]


def sparse_gather_dense_reference(x, *, size, indices):
    del size
    return x[torch.tensor(indices, dtype=torch.int64, device=x.device)]


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
        name="pattern",
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
            self._cached_values[device] = self.values_spec.create_torch_tensor(
                device
            ).clone()
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


class GatherTestCase(TestCase):
    def __str__(self):
        return (
            f"TestCase({self.description} - "
            f"size={self.kwargs['size']}; density={self.kwargs['density']:.6f})"
        )


def parse_test_cases():
    test_cases = []
    for size, density in _TEST_CASES_DATA:
        # 修改: 根据 size 和 density 动态生成索引
        indices = random_spvec_indices(size, density, seed=42)
        nnz = len(indices)
        
        for dtype in _TENSOR_DTYPES:
            values_spec = CachedTensorSpec.from_tensor(
                (nnz,), dtype=dtype, name="values"
            )
            # test_cases.append(
            #     TestCase(
            #         inputs=[
            #             values_spec,
            #             SpVecSpec(values_spec=values_spec, size=size, indices=indices),
            #             TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
            #         ],
            #         kwargs={"size": size, "indices": indices},
            #         tolerance=_TOLERANCE_MAP[dtype],
            #         description=f"SparseGather - OUT_OF_PLACE (size={size})",
            #     )
            # )
            values_spec = CachedTensorSpec.from_tensor(
                (nnz,), dtype=dtype, name="values"
            )
            test_cases.append(
                GatherTestCase(
                    inputs=[
                        values_spec,
                        SpVecSpec(
                            values_spec=values_spec,
                            size=size,
                            indices=indices,
                            mtx_name="sparse_gather",
                            density=density,
                        ),
                        TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "size": size,
                        "density": density,
                        "indices": indices,
                        "out": TensorSpec.from_tensor((nnz,), dtype=dtype, name="out"),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="Gather - OUT(out)",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Gather")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, values, pattern, x, *, size, density, indices, out=None):
        del density
        del values, pattern
        if _use_dense_reference(x.device):
            result = sparse_gather_dense_reference(x, size=size, indices=indices)
        else:
            result = sparse_gather_sparse_reference(x, size=size, indices=indices)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, _values, pattern, x, *, out=None, **_unused):
        return infinicore.sparse_gather(pattern, x, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
    
