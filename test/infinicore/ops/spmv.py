# import sys
# import os

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# import infinicore
# import torch
# from framework import (
#     BaseOperatorTest,
#     TensorSpec,
#     TestCase,
#     GenericTestRunner,
#     is_broadcast,
# )

# # Test cases format: (rows, cols, density)
# _TEST_CASES_DATA = [
#     (24, 36, 0.1),
#     (2048, 2048, 0.0007),
#     (4096, 4096, 0.0001),
# ]

# _TOLERANCE_MAP = {
#     infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
# }

# _TENSOR_DTYPES = [infinicore.float32]


# def generate_unique_indices_batch(nnz, total_elements, device, batch_size=100):
#     """
#     Generate unique random linear indices in [0, total_elements-1] with minimal extra space.
#     Uses a batch approach to avoid excessive memory usage.
#     """
#     generated = set()
#     result = torch.empty(nnz, dtype=torch.long, device=device)
#     count = 0

#     while count < nnz:
#         remaining = nnz - count
#         batch_size = min(batch_size, remaining)
#         candidates = torch.randint(0, total_elements, (batch_size,), device=device)
#         for candidate in candidates.cpu().tolist():
#             if candidate not in generated:
#                 generated.add(candidate)
#                 result[count] = candidate
#                 count += 1
#                 if count == nnz:
#                     break

#     return result


# def create_random_csr_matrix(num_rows, num_cols, density, dtype, device):
#     """
#     Create a random CSR sparse matrix with given density.
#     Returns: values, row_ptr, col_indices, nnz
#     """
#     torch_dtype = dtype  # Assume dtype is already torch dtype
#     torch_device = device  # Assume device is already torch device
#     # Generate random sparse matrix
#     total_elements = num_rows * num_cols
#     nnz = int(total_elements * density)

#     # Generate linear indices for non-zero elements
#     linear_indices = generate_unique_indices_batch(nnz, total_elements, torch_device)

#     rows = linear_indices // num_cols
#     cols = linear_indices % num_cols

#     # Sort by row for CSR format
#     sorted_indices = torch.argsort(rows * num_cols + cols)
#     rows = rows[sorted_indices]
#     cols = cols[sorted_indices]

#     # Create values
#     values = torch.ones(nnz, dtype=torch_dtype, device=torch_device)

#     # Create row pointers (CSR format)
#     row_ptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=torch_device)
#     for i in range(nnz):
#         row_ptr[rows[i] + 1] += 1
#     row_ptr = torch.cumsum(row_ptr, dim=0, dtype=torch.int32)

#     # Column indices
#     col_indices = cols.to(torch.int32).to(torch_device)

#     return values, row_ptr, col_indices, nnz


# def spmv_reference(values, row_ptr, col_indices, x):
#     """
#     Reference SpMV implementation using PyTorch.
#     """
#     num_rows = len(row_ptr) - 1
#     y = torch.zeros(num_rows, dtype=values.dtype, device=values.device)

#     for i in range(num_rows):
#         start = row_ptr[i].item()
#         end = row_ptr[i + 1].item()
#         for j in range(start, end):
#             y[i] += values[j] * x[col_indices[j]]

#     return y


# def parse_test_cases():
#     tests = []
#     for rows, cols, density in _TEST_CASES_DATA:
#         for dtype in _TENSOR_DTYPES:
#             tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
#             # Generate CSR matrix on CPU for spec, but actual test will handle device
#             values, row_ptr, col_indices, nnz = create_random_csr_matrix(rows, cols, density, dtype, torch.device("cpu"))
#             x_shape = (cols,)
#             values_spec = TensorSpec.from_tensor(values.shape, None, dtype)
#             row_ptr_spec = TensorSpec.from_tensor(row_ptr.shape, None, infinicore.int32)
#             col_indices_spec = TensorSpec.from_tensor(col_indices.shape, None, infinicore.int32)
#             x_spec = TensorSpec.from_tensor(x_shape, None, dtype)

#             kwargs = {
#                 "rows": rows,
#                 "cols": cols,
#                 "nnzs": nnz,
#             }

#             tests.append(
#                 TestCase(
#                     inputs=[x_spec, values_spec, row_ptr_spec, col_indices_spec],
#                     kwargs=kwargs,
#                     output_spec=None,
#                     comparison_target=None,
#                     tolerance=tol,
#                     description="SpMV - OUT_OF_PLACE",
#                 )
#             )

#     return tests


# class OpTest(BaseOperatorTest):
#     """SpMV operator test with simplified implementation"""

#     def __init__(self):
#         super().__init__("SpMV")

#     def get_test_cases(self):
#         return parse_test_cases()

#     def torch_operator(self, input_x, values, row_ptr, col_indices, rows, cols, nnzs):
#         # Use reference implementation for comparison
#         return spmv_reference(values, row_ptr, col_indices, input_x)

#     def infinicore_operator(self, input_x, values, row_ptr, col_indices, rows, cols, nnzs):
#         return infinicore.ops.spmv(input_x, values, row_ptr, col_indices, rows, cols, nnzs)


# def main():
#     """Main entry point"""
#     runner = GenericTestRunner(OpTest)
#     runner.run_and_exit()


# if __name__ == "__main__":
#     main()

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

# Test cases format: (rows, cols, density)
_TEST_CASES_DATA = [
    (24, 36, 0.1),
    (1024, 1024, 0.0005),
    (4096, 4096, 0.0001),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float32]

# Mapping from infinicore dtypes to torch dtypes
DTYPE_MAP = {
    infinicore.float32: torch.float32,
    # Add more mappings if needed for other dtypes
}


def generate_unique_indices_batch(nnz, total_elements, device, batch_size=100):
    """
    Generate unique random linear indices in [0, total_elements-1] with minimal extra space.
    Uses a batch approach to avoid excessive memory usage.
    """
    generated = set()
    result = torch.empty(nnz, dtype=torch.long, device=device)
    count = 0

    while count < nnz:
        remaining = nnz - count
        batch_size = min(batch_size, remaining)
        candidates = torch.randint(0, total_elements, (batch_size,), device=device)
        for candidate in candidates.cpu().tolist():
            if candidate not in generated:
                generated.add(candidate)
                result[count] = candidate
                count += 1
                if count == nnz:
                    break

    return result


def create_random_csr_matrix(num_rows, num_cols, density, dtype, device):
    """
    Create a random CSR sparse matrix with given density.
    Returns: values, row_ptr, col_indices, nnz
    """
    torch_dtype = DTYPE_MAP[dtype]  # Map infinicore dtype to torch dtype
    torch_device = device  # Assume device is already torch device
    # Generate random sparse matrix
    total_elements = num_rows * num_cols
    nnz = int(total_elements * density)

    # Generate linear indices for non-zero elements
    linear_indices = generate_unique_indices_batch(nnz, total_elements, torch_device)

    rows = linear_indices // num_cols
    cols = linear_indices % num_cols

    # Sort by row for CSR format
    sorted_indices = torch.argsort(rows * num_cols + cols)
    rows = rows[sorted_indices]
    cols = cols[sorted_indices]

    # Create values
    values = torch.ones((nnz,), dtype=torch_dtype, device=torch_device)

    # Create row pointers (CSR format)
    row_ptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=torch_device)
    for i in range(nnz):
        row_ptr[rows[i] + 1] += 1
    row_ptr = torch.cumsum(row_ptr, dim=0, dtype=torch.int32)

    # Column indices
    col_indices = cols.to(torch.int32).to(torch_device)

    return values, row_ptr, col_indices, nnz


def spmv_reference(values, row_ptr, col_indices, x):
    """
    Reference SpMV implementation using PyTorch.
    """
    num_rows = len(row_ptr) - 1
    y = torch.zeros(num_rows, dtype=values.dtype, device=values.device)

    for i in range(num_rows):
        start = row_ptr[i].item()
        end = row_ptr[i + 1].item()
        for j in range(start, end):
            y[i] += values[j] * x[col_indices[j]]

    return y


def parse_test_cases():
    tests = []
    for rows, cols, density in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            # Generate CSR matrix on CPU for spec, but actual test will handle device
            values, row_ptr, col_indices, nnz = create_random_csr_matrix(rows, cols, density, dtype, torch.device("cpu"))
            x_shape = (cols,)
            values_spec = TensorSpec.from_tensor(values.shape, None, dtype)
            row_ptr_spec = TensorSpec.from_tensor(row_ptr.shape, None, infinicore.int32)
            col_indices_spec = TensorSpec.from_tensor(col_indices.shape, None, infinicore.int32)
            x_spec = TensorSpec.from_tensor(x_shape, None, dtype)

            kwargs = {
                "rows": rows,
                "cols": cols,
                "nnzs": nnz,
            }

            tests.append(
                TestCase(
                    inputs=[x_spec, values_spec, row_ptr_spec, col_indices_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="SpMV - OUT_OF_PLACE",
                )
            )

    return tests


class OpTest(BaseOperatorTest):
    """SpMV operator test with simplified implementation"""

    def __init__(self):
        super().__init__("SpMV")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input_x, values, row_ptr, col_indices, rows, cols, nnzs):
        # Use reference implementation for comparison
        return spmv_reference(values, row_ptr, col_indices, input_x)

    def infinicore_operator(self, input_x, values, row_ptr, col_indices, rows, cols, nnzs):
        return infinicore.ops.spmv(input_x, values, row_ptr, col_indices, rows, cols, nnzs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()