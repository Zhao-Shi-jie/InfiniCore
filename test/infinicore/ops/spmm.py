import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch


def test_spmm_cpu():
    device = infinicore.device("cpu", 0)
    crow = infinicore.from_list([0, 2, 3, 5], dtype=infinicore.int32, device=device)
    col = infinicore.from_list([0, 2, 1, 0, 3], dtype=infinicore.int32, device=device)
    values_torch = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    b_torch = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    )

    values = infinicore.from_torch(values_torch)
    b = infinicore.from_torch(b_torch)
    sparse = infinicore.csr_spmat(crow, col, values, (3, 4))

    out = infinicore.gemm(sparse, b)
    expected = torch.sparse_csr_tensor(
        torch.tensor([0, 2, 3, 5], dtype=torch.int32),
        torch.tensor([0, 2, 1, 0, 3], dtype=torch.int32),
        values_torch,
        size=(3, 4),
    ).matmul(b_torch)

    actual = torch.empty_like(expected)
    actual_infinicore = infinicore.from_torch(actual)
    actual_infinicore.copy_(out)
    assert torch.allclose(actual, expected)


if __name__ == "__main__":
    test_spmm_cpu()
    print("\033[92mTest passed!\033[0m")
