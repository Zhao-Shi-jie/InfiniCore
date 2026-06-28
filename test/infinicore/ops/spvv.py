import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from framework.utils.tensor_utils import infinicore_tensor_from_torch
from sparse_utils import infinicore_list_on_device


def _summarize_indices(indices, limit=6):
    if len(indices) <= limit:
        return str(indices)
    head = ", ".join(str(v) for v in indices[: limit // 2])
    tail = ", ".join(str(v) for v in indices[-(limit // 2) :])
    return f"[{head}, ..., {tail}]"


class SparseTestCase(TestCase):
    def __str__(self):
        input_str = "; ".join(str(inp) for inp in self.inputs)
        size = self.kwargs["size"]
        indices = self.kwargs["indices"]
        kwargs_strs = [
            f"size={size}",
            f"indices={_summarize_indices(indices)}",
        ]
        out = self.kwargs.get("out")
        if out is not None:
            kwargs_strs.append(f"out={out}")
        return (
            f"TestCase({self.description} - inputs=[{input_str}], "
            f"kwargs={{{'; '.join(kwargs_strs)}}})"
        )


def _generate_spvv_cases():
    cases = []
    random.seed(42)
    # (size, density)
    configs = [
        # (128, 0.01),
        # (1024, 0.01),
        # (4096, 0.01),
        (40960000, 0.01)
    ]
    for size, density in configs:
        nnz = int(size * density)
        indices = sorted(random.sample(range(size), nnz))
        cases.append((size, indices))
    return cases

_TEST_CASES_DATA = _generate_spvv_cases()

# _TEST_CASES_DATA = [
#     (6, [0, 2, 5]),
#     (8, [1, 3, 4, 7]),
# ]

_TOLERANCE_MAP = {
    # infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-4},
    # infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
    # infinicore.float32: {"atol": 1e-3, "rtol": 1e-3},
}

_TENSOR_DTYPES = [
    # infinicore.float16,
    # infinicore.bfloat16,
    infinicore.float32,
]


def _use_dense_reference(device):
    return device.type == "mlu"


def spvv_sparse_reference(values, x, *, size, indices):
    indices_tensor = torch.tensor(indices, dtype=torch.int64, device=values.device)
    sparse = torch.sparse_coo_tensor(
        indices_tensor.unsqueeze(0),
        values,
        size=(size,),
        device=values.device,
    )
    return torch.dot(sparse.to_dense(), x)


def spvv_dense_reference(values, x, *, size, indices):
    sparse_dense = torch.zeros(size, dtype=values.dtype, device=values.device)
    sparse_dense[torch.tensor(indices, dtype=torch.int64, device=values.device)] = values
    return torch.dot(sparse_dense, x)


class SpVecSpec(TensorSpec):
    def __init__(self, *, values_spec, size, indices, name="sparse"):
        super().__init__(shape=(size,), dtype=values_spec.dtype, name=name)
        self.values_spec = values_spec
        self.size = size
        self.indices = indices
        self._cached_values = None

    def create_torch_tensor(self, device):
        if self._cached_values is None:
            self._cached_values = self.values_spec.create_torch_tensor(device)
        values = self._cached_values
        indices_tensor = infinicore_list_on_device(
            self.indices,
            dtype=infinicore.int32,
            device=infinicore_tensor_from_torch(values).device,
        )
        return infinicore.coo_spvec(
            indices_tensor,
            infinicore_tensor_from_torch(values),
            self.size,
        )

    def __str__(self):
        return f"{self.name}: spvec(size={self.size})"


class CachedTensorSpec(TensorSpec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    @classmethod
    def from_tensor(
        cls,
        shape,
        strides=None,
        dtype=None,
        init_mode=None,
        **kwargs,
    ):
        if init_mode is None:
            return cls(shape=shape, dtype=dtype, strides=strides, **kwargs)
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            init_mode=init_mode,
            **kwargs,
        )

    def create_torch_tensor(self, device):
        if device not in self._cache:
            self._cache[device] = super().create_torch_tensor(device)
        return self._cache[device]


def parse_test_cases():
    test_cases = []
    for size, indices in _TEST_CASES_DATA:
        nnz = len(indices)
        for dtype in _TENSOR_DTYPES:
            values_spec = CachedTensorSpec.from_tensor((nnz,), dtype=dtype, name="values")
            # test_cases.append(
            #     SparseTestCase(
            #         inputs=[
            #             values_spec,
            #             SpVecSpec(
            #                 values_spec=values_spec,
            #                 size=size,
            #                 indices=indices,
            #             ),
            #             TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
            #         ],
            #         kwargs={
            #             "size": size,
            #             "indices": indices,
            #         },
            #         tolerance=_TOLERANCE_MAP[dtype],
            #         description="SpVV - OUT_OF_PLACE",
            #     )
            # )
            values_spec = CachedTensorSpec.from_tensor((nnz,), dtype=dtype, name="values")
            test_cases.append(
                SparseTestCase(
                    inputs=[
                        values_spec,
                        SpVecSpec(
                            values_spec=values_spec,
                            size=size,
                            indices=indices,
                        ),
                        TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "size": size,
                        "indices": indices,
                        "out": TensorSpec.from_tensor((), dtype=dtype, name="out"),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SpVV - OUT(out)",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SpVV")

    def get_test_cases(self):
        return parse_test_cases()

    # def torch_operator(self, values, sparse, x, *, size, indices, out=None):
    #     del sparse
    #     if _use_dense_reference(values.device):
    #         result = spvv_dense_reference(values, x, size=size, indices=indices)
    #     else:
    #         result = spvv_sparse_reference(values, x, size=size, indices=indices)
    #     if out is not None:
    #         out.copy_(result)
    #         return out
    #     return result

    def infinicore_operator(self, _values, sparse, x, *, size, indices, out=None):
        return infinicore.spvv(sparse, x, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
