import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_size_t, c_void_p, c_float
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    InfiniDtype,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # num_rows, num_cols, density
    (100, 100, 0.01),      # Dense small matrix
    (800, 1000, 0.01),
    (1000, 1000, 0.01),  # Small sparse matrix
    (500, 800, 0.02),    # Non-square matrix
    (2048, 2048, 0.005), # Larger matrix, very sparse
    (100, 200, 0.1),     # Dense small matrix
    (1500, 1200, 0.015), # Medium size
]

# Data types used for testing (currently only float32 supported)
_TENSOR_DTYPES = [torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100

# ==============================================================================
#  Definitions
# ==============================================================================
class SpMVDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopSpMVDescriptor_t = POINTER(SpMVDescriptor)

def create_random_csr_matrix(num_rows, num_cols, density, dtype, device):
    """
    Create a random CSR sparse matrix with given density.
    Returns: values, row_ptr, col_indices, nnz
    """
    # Generate random sparse matrix
    total_elements = num_rows * num_cols
    nnz = int(total_elements * density)
    
    # Generate random indices and sort them to avoid duplicates
    linear_indices = torch.randperm(total_elements)[:nnz]
    rows = linear_indices // num_cols
    cols = linear_indices % num_cols
    
    # Sort by row for CSR format
    sorted_indices = torch.argsort(rows * num_cols + cols)
    rows = rows[sorted_indices]
    cols = cols[sorted_indices]
    
    # Create values
    values = torch.ones(nnz, dtype=dtype, device=device)
    
    # Create row pointers (CSR format)
    row_ptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
    for i in range(nnz):
        row_ptr[rows[i] + 1] += 1
    row_ptr = torch.cumsum(row_ptr, dim=0, dtype=torch.int32)
    # # 🔧 修复：强制row_ptr连续
    # if not row_ptr.is_contiguous():
    #     print("Warning: row_ptr from cumsum is not contiguous!")
    #     row_ptr = row_ptr.contiguous()
    
    # Column indices
    col_indices = cols.to(torch.int32).to(device)
    # # 🔧 修复：强制col_indices连续（虽然通常已经连续）
    # if not col_indices.is_contiguous():
    #     print("Warning: col_indices is not contiguous!")
    #     col_indices = col_indices.contiguous()
    
    if DEBUG:
        print("=== CSR Matrix Memory Layout Debug ===")
        print(f"row_ptr: shape={row_ptr.shape}, dtype={row_ptr.dtype}")
        print(f"row_ptr.is_contiguous(): {row_ptr.is_contiguous()}")
        print(f"row_ptr.stride(): {row_ptr.stride()}")
        print(f"row_ptr.storage_offset(): {row_ptr.storage_offset()}")
        print(f"row_ptr values: {row_ptr[:10]}")
        
        print(f"col_indices: shape={col_indices.shape}, dtype={col_indices.dtype}")
        print(f"col_indices.is_contiguous(): {col_indices.is_contiguous()}")
        print(f"col_indices.stride(): {col_indices.stride()}")
        print(f"col_indices.storage_offset(): {col_indices.storage_offset()}")
        print(f"col_indices values: {col_indices[:10]}")

    return values, row_ptr, col_indices, nnz

def spmv_reference(values, row_ptr, col_indices, x):
    """
    Reference SpMV implementation using PyTorch.
    """
    # 打印values, row_ptr, col_indices, x for debugging
    if DEBUG:
        print("------------------SpMV Reference Parameters:--------------------")
        print("Values:", values)
        print("Row pointers:", row_ptr)
        print("Column indices:", col_indices)
        print("Input vector x:", x)
    num_rows = len(row_ptr) - 1
    y = torch.zeros(num_rows, dtype=values.dtype, device=values.device)
    
    for i in range(num_rows):
        start = row_ptr[i].item()
        end = row_ptr[i + 1].item()
        for j in range(start, end):
            y[i] += values[j] * x[col_indices[j]]
    
    return y

def spmv_pytorch_reference(values, row_ptr, col_indices, x, num_rows, num_cols):
    """
    Alternative reference using PyTorch sparse tensors for verification.
    """
    # Convert CSR to COO format for PyTorch sparse tensor
    row_indices = []
    for i in range(num_rows):
        start = row_ptr[i].item()
        end = row_ptr[i + 1].item()
        row_indices.extend([i] * (end - start))
    
    row_indices = torch.tensor(row_indices, dtype=torch.long, device=values.device)
    col_indices_long = col_indices.long()
    
    # Create sparse tensor
    indices = torch.stack([row_indices, col_indices_long])
    sparse_matrix = torch.sparse_coo_tensor(
        indices, values, (num_rows, num_cols), device=values.device
    ).coalesce()
    
    # Perform SpMV
    return torch.sparse.mm(sparse_matrix, x.unsqueeze(1)).squeeze(1)

# The argument list should be (lib, handle, torch_device, <param list>, dtype)
def test(
    lib,
    handle,
    torch_device,
    num_rows,
    num_cols,
    density,
    dtype=torch.float32,
    sync=None
):
    print(
        f"Testing SpMV on {torch_device} with num_rows:{num_rows}, num_cols:{num_cols}, "
        f"density:{density}, dtype:{dtype}"
    )

    # Create random CSR sparse matrix
    values, row_ptr, col_indices, nnz = create_random_csr_matrix(
        num_rows, num_cols, density, dtype, torch_device
    )
    
    # Create input vector
    x = torch.ones(num_cols, dtype=dtype, device=torch_device)
    
    # Create output vector
    y = torch.zeros(num_rows, dtype=dtype, device=torch_device)

    # Compute reference results
    y_ref = spmv_reference(values, row_ptr, col_indices, x)
    y_pytorch_ref = spmv_pytorch_reference(values, row_ptr, col_indices, x, num_rows, num_cols)
    
    # Verify our reference implementations agree
    assert torch.allclose(y_ref, y_pytorch_ref, atol=1e-6, rtol=1e-5), \
        "Reference implementations don't match!"

    # Create tensors for infiniop
    y_tensor = to_tensor(y, lib)
    x_tensor = to_tensor(x, lib)
    values_tensor = to_tensor(values, lib)
    row_ptr_tensor = to_tensor(row_ptr, lib)
    col_indices_tensor = to_tensor(col_indices, lib)

    if sync is not None:
        sync()

    # Create descriptor
    descriptor = infiniopSpMVDescriptor_t()
    check_error(
        lib.infiniopCreateSpMVDescriptor(
            handle,
            ctypes.byref(descriptor),
            num_cols,
            num_rows,
            nnz,
            InfiniDtype.F32 # Only support float32 now.
        )
    )

    # Invalidate the descriptors to prevent them from being directly used by the kernel
    for tensor in [y_tensor, x_tensor, values_tensor, row_ptr_tensor, col_indices_tensor]:
        tensor.destroyDesc(lib)

    # Execute infiniop SpMV operator
    def lib_spmv():
        check_error(
            lib.infiniopSpMV(
                descriptor,
                y_tensor.data,
                x_tensor.data,
                values_tensor.data,
                row_ptr_tensor.data,
                col_indices_tensor.data,
                None,  # stream
            )
        )
    
    if DEBUG:# 打印spmv计算需要的参数，x_tensor, y_tensor, values_tensor, row_ptr_tensor, col_indices_tensor
        print("--------------SpMV parameters(数值): ------------------")
        print("x_tensor:", x_tensor.torch_tensor_)
        print("y_tensor:", y_tensor.torch_tensor_)
        print("values_tensor:", values_tensor.torch_tensor_)
        print("row_ptr_tensor:", row_ptr_tensor.torch_tensor_)
        print("col_indices_tensor:", col_indices_tensor.torch_tensor_)

        print("--------------SpMV parameters(地址): ------------------")
        print("x_tensor:", x_tensor.data)
        print("y_tensor:", y_tensor.data)
        print("values_tensor:", values_tensor.data)
        print("row_ptr_tensor:", row_ptr_tensor.data)   
        print("col_indices_tensor:", col_indices_tensor.data)

    lib_spmv()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    # if DEBUG:
    #     debug(y, y_ref, atol=atol, rtol=rtol)
    
    # Check against our reference
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Results don't match reference! Max diff: {(y - y_ref).abs().max().item()}"
    
    # Also check against PyTorch sparse reference
    assert torch.allclose(y, y_pytorch_ref, atol=atol, rtol=rtol), \
        f"Results don't match PyTorch reference! Max diff: {(y - y_pytorch_ref).abs().max().item()}"

    # Profiling workflow
    if PROFILE:
        profile_operation(
            "Reference", 
            lambda: spmv_reference(values, row_ptr, col_indices, x), 
            torch_device, NUM_PRERUN, NUM_ITERATIONS
        )
        profile_operation(
            "PyTorch", 
            lambda: spmv_pytorch_reference(values, row_ptr, col_indices, x, num_rows, num_cols),
            torch_device, NUM_PRERUN, NUM_ITERATIONS
        )
        profile_operation(
            "    lib", 
            lambda: lib_spmv(), 
            torch_device, NUM_PRERUN, NUM_ITERATIONS
        )
    
    check_error(lib.infiniopDestroySpMVDescriptor(descriptor))

# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    # Register API functions
    lib.infiniopCreateSpMVDescriptor.restype = c_int32
    lib.infiniopCreateSpMVDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSpMVDescriptor_t),
        c_size_t,  # num_cols
        c_size_t,  # num_rows  
        c_size_t,  # nnz
        c_int32,   # dtype
    ]

    lib.infiniopSpMV.restype = c_int32
    lib.infiniopSpMV.argtypes = [
        infiniopSpMVDescriptor_t,
        c_void_p,  # y
        c_void_p,  # x
        c_void_p,  # values
        c_void_p,  # row_ptr
        c_void_p,  # col_indices
        c_void_p,  # stream
    ]

    lib.infiniopDestroySpMVDescriptor.restype = c_int32
    lib.infiniopDestroySpMVDescriptor.argtypes = [
        infiniopSpMVDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
