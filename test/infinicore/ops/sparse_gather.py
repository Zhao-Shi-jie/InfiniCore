# import os
# import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# import infinicore
# import torch
# from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
# from framework.utils.tensor_utils import infinicore_tensor_from_torch

# _TEST_CASES_DATA = [
#     # size, indices
#     (256, [0, 2, 5, 9, 44, 66, 23, 123, 200]),
#     (4096, [7, 1, 3, 0, 1023, 2047, 4095, 256, 512, 1024, 2048, 3000, 3500]),
#     (10000, [10, 50, 200, 500, 999, 1234, 4321, 5678, 6789, 7890]),
# ]

# _TENSOR_DTYPES = [infinicore.float32]

# _TOLERANCE_MAP = {
#     infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
# }


# def _use_dense_reference(device):
#     return device.type == "mlu"


# def sparse_gather_sparse_reference(x, *, size, indices):
#     indices_tensor = torch.tensor(indices, dtype=torch.int64, device=x.device)
#     values = torch.ones(len(indices), dtype=x.dtype, device=x.device)
#     torch.sparse_coo_tensor(
#         indices_tensor.unsqueeze(0),
#         values,
#         size=(size,),
#         device=x.device,
#     )
#     return x[indices_tensor]


# def sparse_gather_dense_reference(x, *, size, indices):
#     del size
#     return x[torch.tensor(indices, dtype=torch.int64, device=x.device)]


# class CachedTensorSpec(TensorSpec):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._cache = {}

#     @classmethod
#     def from_tensor(cls, shape, strides=None, dtype=None, init_mode=None, **kwargs):
#         if init_mode is None:
#             return cls(shape=shape, dtype=dtype, strides=strides, **kwargs)
#         return cls(
#             shape=shape, dtype=dtype, strides=strides, init_mode=init_mode, **kwargs
#         )

#     def create_torch_tensor(self, device):
#         if device not in self._cache:
#             self._cache[device] = super().create_torch_tensor(device)
#         return self._cache[device]


# class SpVecSpec(TensorSpec):
#     def __init__(self, *, values_spec, size, indices, name="pattern"):
#         super().__init__(shape=(size,), dtype=values_spec.dtype, name=name)
#         self.values_spec = values_spec
#         self.size = size
#         self.indices = indices
#         self._cached_values = {}

#     def create_torch_tensor(self, device):
#         if device not in self._cached_values:
#             self._cached_values[device] = self.values_spec.create_torch_tensor(
#                 device
#             ).clone()
#         values = self._cached_values[device]
#         infini_values = infinicore_tensor_from_torch(values)
#         indices_tensor = infinicore.from_list(
#             self.indices, dtype=infinicore.int64, device=infini_values.device
#         )
#         return infinicore.coo_spvec(indices_tensor, infini_values, self.size)

#     def __str__(self):
#         density = len(self.indices) / self.size if self.size else 0
#         return f"{self.name}: spvec(size={self.size})"


# def parse_test_cases():
#     test_cases = []
#     for size, indices in _TEST_CASES_DATA:
#         nnz = len(indices)
#         for dtype in _TENSOR_DTYPES:
#             values_spec = CachedTensorSpec.from_tensor(
#                 (nnz,), dtype=dtype, name="values"
#             )
#             # test_cases.append(
#             #     TestCase(
#             #         inputs=[
#             #             values_spec,
#             #             SpVecSpec(values_spec=values_spec, size=size, indices=indices),
#             #             TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
#             #         ],
#             #         kwargs={"size": size, "indices": indices},
#             #         tolerance=_TOLERANCE_MAP[dtype],
#             #         description="SparseGather - OUT_OF_PLACE",
#             #     )
#             # )
#             values_spec = CachedTensorSpec.from_tensor(
#                 (nnz,), dtype=dtype, name="values"
#             )
#             test_cases.append(
#                 TestCase(
#                     inputs=[
#                         values_spec,
#                         SpVecSpec(values_spec=values_spec, size=size, indices=indices),
#                         TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
#                     ],
#                     kwargs={
#                         "size": size,
#                         "indices": indices,
#                         "out": TensorSpec.from_tensor((nnz,), dtype=dtype, name="out"),
#                     },
#                     comparison_target="out",
#                     tolerance=_TOLERANCE_MAP[dtype],
#                     description="SparseGather - OUT(out)",
#                 )
#             )
#     return test_cases


# class OpTest(BaseOperatorTest):
#     def __init__(self):
#         super().__init__("SparseGather")

#     def get_test_cases(self):
#         return parse_test_cases()

#     def torch_operator(self, values, pattern, x, *, size, indices, out=None):
#         del values, pattern
#         if _use_dense_reference(x.device):
#             result = sparse_gather_dense_reference(x, size=size, indices=indices)
#         else:
#             result = sparse_gather_sparse_reference(x, size=size, indices=indices)
#         if out is not None:
#             out.copy_(result)
#             return out
#         return result

#     def infinicore_operator(self, _values, pattern, x, *, size, indices, out=None):
#         return infinicore.sparse_gather(pattern, x, out=out)


# if __name__ == "__main__":
#     runner = GenericTestRunner(OpTest)
#     runner.run_and_exit()


import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from framework.utils.tensor_utils import infinicore_tensor_from_torch

# 修改: 测试用例改为 (size, density) 格式
# density 表示稀疏度，例如 0.01 表示 1% 的非零元素
_TEST_CASES_DATA = [
    # size, density
    (128, 0.01),      # 小尺寸，较高密度
    (1024, 0.01),
    # (2048, 0.01),     # 中等尺寸，低密度
    (4096, 0.01),      # 中等尺寸，较高密度
    (81920000, 0.01),   # 大尺寸，极低密度
]

_TENSOR_DTYPES = [infinicore.float32]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}

# 固定随机种子以保证测试可复现
_RANDOM_SEED = 42


def _generate_indices(size, density, device="cpu"):
    """
    根据给定的 size 和 density 生成随机且不重复的索引。
    返回排序后的索引列表（COO 格式通常需要有序索引）。
    """
    nnz = max(1, int(size * density))  # 至少保证有1个非零元素
    nnz = min(nnz, size)  # 不能超过总大小
    
    generator = torch.Generator(device=device)
    generator.manual_seed(_RANDOM_SEED)
    
    # 使用 randperm 生成不重复的随机索引
    indices = torch.randperm(size, generator=generator, device=device)[:nnz]
    # COO 格式通常要求索引有序
    indices, _ = torch.sort(indices)
    
    return indices.tolist()


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
    def __init__(self, *, values_spec, size, indices, name="pattern"):
        super().__init__(shape=(size,), dtype=values_spec.dtype, name=name)
        self.values_spec = values_spec
        self.size = size
        self.indices = indices
        self._cached_values = {}

    def create_torch_tensor(self, device):
        if device not in self._cached_values:
            self._cached_values[device] = self.values_spec.create_torch_tensor(
                device
            ).clone()
        values = self._cached_values[device]
        infini_values = infinicore_tensor_from_torch(values)
        indices_tensor = infinicore.from_list(
            self.indices, dtype=infinicore.int64, device=infini_values.device
        )
        return infinicore.coo_spvec(indices_tensor, infini_values, self.size)

    def __str__(self):
        return f"{self.name}: spvec(size={self.size})"


def parse_test_cases():
    test_cases = []
    for size, density in _TEST_CASES_DATA:
        # 修改: 根据 size 和 density 动态生成索引
        indices = _generate_indices(size, density)
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
                TestCase(
                    inputs=[
                        values_spec,
                        SpVecSpec(values_spec=values_spec, size=size, indices=indices),
                        TensorSpec.from_tensor((size,), dtype=dtype, name="x"),
                    ],
                    kwargs={
                        "size": size,
                        "indices": indices,
                        "out": TensorSpec.from_tensor((nnz,), dtype=dtype, name="out"),
                    },
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description=f"SparseGather - OUT(out) (size={size})",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SparseGather")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, values, pattern, x, *, size, indices, out=None):
        del values, pattern
        if _use_dense_reference(x.device):
            result = sparse_gather_dense_reference(x, size=size, indices=indices)
        else:
            result = sparse_gather_sparse_reference(x, size=size, indices=indices)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, _values, pattern, x, *, size, indices, out=None):
        return infinicore.sparse_gather(pattern, x, out=out)


if __name__ == "__main__":
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()
    
