from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64, c_bool, c_int
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor
)

from operatorspy.tests.test_utils import get_args
import torch
from typing import Tuple
import numpy as np

# For CSR format:
# row_indices = row pointers
# values = non-zero values
# col_indices = column indices
class SpMVDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopSpMVDescriptor_t = POINTER(SpMVDescriptor)

def spmv_reference(values, row_indices, col_indices, x, sparse_format=0):
    """
    Compute the SpMV reference result on the CPU.
    sparse_format: 0 for CSR, 1 for COO
    """
    if sparse_format == 0:  # CSR
        num_rows = len(row_indices) - 1
        y = torch.zeros(num_rows, dtype=values.dtype, device=values.device)
        
        for i in range(num_rows):
            for j in range(row_indices[i], row_indices[i + 1]):
                y[i] += values[j] * x[col_indices[j]]
    else:  # COO
        num_rows = int(row_indices.max().item()) + 1
        y = torch.zeros(num_rows, dtype=values.dtype, device=values.device)
        
        for i in range(len(values)):
            y[row_indices[i]] += values[i] * x[col_indices[i]]
    
    return y

def test(
    lib,
    handle,
    torch_device,
    num_rows,
    num_cols,
    density,
    sparse_format=0,  # 0 for CSR, 1 for COO
    dtype=torch.float32,
    sync=None
):
    print(
        f"Testing SpMV on {torch_device} with num_rows:{num_rows} num_cols:{num_cols} "
        f"density:{density} sparse_format:{sparse_format} dtype:{dtype}"
    )
    
    # Create random sparse matrix in CSR/COO format
    nnz = int(num_rows * num_cols * density)
    
    if sparse_format == 0:  # CSR format
        # Create random CSR matrix
        indices = torch.randint(0, num_rows * num_cols, (nnz,))
        unique_indices = torch.unique(indices)
        nnz_actual = len(unique_indices)
        
        # Convert linear indices to row, col
        rows = unique_indices // num_cols
        cols = unique_indices % num_cols
        
        # Sort by row then by column for CSR
        sorted_indices = torch.argsort(rows * num_cols + cols)
        rows = rows[sorted_indices]
        cols = cols[sorted_indices]
        
        # Create values
        values = torch.rand(nnz_actual, dtype=dtype).to(torch_device)
        if dtype == torch.float16:
            values = (values * 10).to(dtype)  # Scale for better numerical stability
        else:
            values = values.to(dtype)
        
        # Create row pointers
        row_indices = torch.zeros(num_rows + 1, dtype=torch.int32).to(torch_device)
        for r in rows:
            row_indices[r + 1] += 1
        row_indices = torch.cumsum(row_indices, dim=0).to(torch.int32)
        
        # Column indices
        col_indices = cols.to(torch.int32).to(torch_device)
        
    else:  # COO format
        # Random row and column indices
        row_indices = torch.randint(0, num_rows, (nnz,), dtype=torch.int32).to(torch_device)
        col_indices = torch.randint(0, num_cols, (nnz,), dtype=torch.int32).to(torch_device)
        values = torch.rand(nnz, dtype=dtype).to(torch_device)
        if dtype == torch.float16:
            values = (values * 10).to(dtype)
        else:
            values = values.to(dtype)
    
    # Create input vector
    x = torch.rand(num_cols, dtype=dtype).to(torch_device)
    if dtype == torch.float16:
        x = (x * 10).to(dtype)
    else:
        x = x.to(dtype)
    
    # Compute reference result
    with torch.no_grad():
        y_ref = spmv_reference(values, row_indices, col_indices, x, sparse_format)
    
    # Create output vector
    y = torch.zeros(num_rows, dtype=dtype).to(torch_device)
    
    # Create tensors
    y_tensor = to_tensor(y, lib)
    x_tensor = to_tensor(x, lib)
    values_tensor = to_tensor(values, lib)
    row_indices_tensor = to_tensor(row_indices, lib)
    col_indices_tensor = to_tensor(col_indices, lib)
    
    if sync is not None:
        sync()
    
    # Create descriptor
    descriptor = infiniopSpMVDescriptor_t()
    check_error(
        lib.infiniopCreateSpMVDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            values_tensor.descriptor,
            row_indices_tensor.descriptor,
            col_indices_tensor.descriptor,
            c_int(sparse_format)
        )
    )
    
    # Invalidate descriptors to ensure kernel uses proper values
    y_tensor.descriptor.contents.invalidate()
    x_tensor.descriptor.contents.invalidate()
    values_tensor.descriptor.contents.invalidate()
    row_indices_tensor.descriptor.contents.invalidate()
    col_indices_tensor.descriptor.contents.invalidate()
    
    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetSpMVWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    
    # Create workspace
    if workspace_size.value > 0:
        workspace = torch.zeros(int(workspace_size.value), dtype=torch.uint8).to(torch_device)
        workspace_ptr = workspace.data_ptr()
    else:
        workspace_ptr = None
    
    # Execute SpMV
    check_error(
        lib.infiniopSpMV(
            descriptor,
            workspace_ptr,
            workspace_size.value,
            y_tensor.data,
            x_tensor.data,
            values_tensor.data,
            row_indices_tensor.data,
            col_indices_tensor.data,
            None
        )
    )
    
    # Check results
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(y, y_ref, rtol=rtol, atol=atol), \
           f"Results don't match! Max diff: {(y - y_ref).abs().max().item()}"
    
    # Destroy descriptor
    check_error(lib.infiniopDestroySpMVDescriptor(descriptor))
    
    print("Test passed!")


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    
    for num_rows, num_cols, density, sparse_format, dtype in test_cases:
        test(lib, handle, "cpu", num_rows, num_cols, density, sparse_format, dtype)
    
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
        
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    
    for num_rows, num_cols, density, sparse_format, dtype in test_cases:
        test(lib, handle, "cuda", num_rows, num_cols, density, sparse_format, dtype)
    
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # num_rows, num_cols, density, sparse_format, dtype
        (1000, 1000, 0.01, 0, torch.float32),  # CSR, float32
        (500, 800, 0.02, 0, torch.float16),    # CSR, float16
        (800, 500, 0.01, 1, torch.float32),    # COO, float32
        (400, 600, 0.02, 1, torch.float16),    # COO, float16
    ]
    
    args = get_args()
    lib = open_lib()
    
    # Register API functions
    lib.infiniopCreateSpMVDescriptor.restype = c_int32
    lib.infiniopCreateSpMVDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSpMVDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int
    ]
    
    lib.infiniopGetSpMVWorkspaceSize.restype = c_int32
    lib.infiniopGetSpMVWorkspaceSize.argtypes = [
        infiniopSpMVDescriptor_t,
        POINTER(c_uint64)
    ]
    
    lib.infiniopSpMV.restype = c_int32
    lib.infiniopSpMV.argtypes = [
        infiniopSpMVDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p
    ]
    
    lib.infiniopDestroySpMVDescriptor.restype = c_int32
    lib.infiniopDestroySpMVDescriptor.argtypes = [
        infiniopSpMVDescriptor_t
    ]
    
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    
    print("\033[92mAll tests passed!\033[0m")
