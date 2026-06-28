import random

import infinicore
import torch
from framework import TensorSpec
from framework.datatypes import to_torch_dtype
from framework.devices import torch_device_map


def density_nnz(total, density):
    if total <= 0:
        raise ValueError("total element count must be positive")
    nnz = int(total * density + 0.5)
    return min(total, max(1, nnz))


def random_csr_indices(rows, cols, density, *, seed):
    total = rows * cols
    nnz = density_nnz(total, density)
    rng = random.Random(seed)
    positions = sorted(rng.sample(range(total), nnz))
    crow = [0] * (rows + 1)
    col = []
    for position in positions:
        row = position // cols
        crow[row + 1] += 1
        col.append(position % cols)
    for row in range(rows):
        crow[row + 1] += crow[row]
    return crow, col


def random_spvec_indices(size, density, *, seed):
    nnz = density_nnz(size, density)
    rng = random.Random(seed)
    return sorted(rng.sample(range(size), nnz))


def random_values(nnz, *, seed):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(nnz)]


def infinicore_list_on_device(data, *, dtype, device):
    return infinicore.from_list(data, dtype=dtype).to(device)


class ValuesFromListSpec(TensorSpec):
    def __init__(self, values, *, dtype, name="values"):
        super().__init__(shape=(len(values),), dtype=dtype, name=name)
        self.values = list(values)

    def create_torch_tensor(self, device):
        return torch.tensor(
            self.values,
            dtype=to_torch_dtype(self.dtype),
            device=torch_device_map[device],
        )
