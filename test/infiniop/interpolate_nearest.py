import torch
import ctypes
from ctypes import c_uint64

from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto
from typing import List, Tuple
import math
from torch.nn import functional as F

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

# Test cases: (input_shape, input_stride, output_shape, output_stride)
_TEST_CASES = [
    # Simple test case with contiguous strides first
    (
        (1, 1, 2, 2),
        None,  # Use default contiguous strides
        (1, 1, 4, 4),
        None,  # Use default contiguous strides
    ),
    # Simple upscaling 2x
    (
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        (1, 3, 8, 8),
        (192, 64, 8, 1),
    ),
    # Simple downscaling 2x
    (
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        (1, 3, 4, 4),
        (48, 16, 4, 1),
    ),
    # Batch upscaling
    (
        (2, 4, 2, 2),
        (16, 4, 2, 1),
        (2, 4, 6, 6),
        (144, 36, 6, 1),
    ),
    # Different aspect ratio scaling
    (
        (1, 1, 3, 5),
        (15, 15, 5, 1),
        (1, 1, 9, 10),
        (90, 90, 10, 1),
    ),
    # Large batch, more channels
    (
        (4, 64, 16, 16),
        (16384, 256, 16, 1),
        (4, 64, 32, 32),
        (65536, 1024, 32, 1),
    ),
    # Small to large upscaling
    (
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 7, 7),
        (49, 49, 7, 1),
    ),
    # 测例1: 非连续内存布局 - Channel-first 转置
    (
        (1, 2, 3, 4),    # input_shape: N=1, C=2, H=3, W=4
        (24, 1, 8, 2),   # input_stride: 特殊stride，C维stride=1（最小），类似转置
        (1, 2, 6, 8),    # output_shape
        (96, 1, 16, 2),  # output_stride: 保持相同的stride模式
    ),

    # 测例2: 带有内存间隙的布局 - Padded strides
    (
        (2, 3, 2, 2),    # input_shape: N=2, C=3, H=2, W=2  
        (32, 8, 4, 1),   # input_stride: 每个维度都有额外的padding间隙
        (2, 3, 4, 4),    # output_shape
        (128, 32, 8, 1), # output_stride: 保持padding比例
    ),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.I8]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.I8: {"atol": 1, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def interpolate_nearest(input_tensor, output_shape, output_tensor):
    """
    Perform nearest neighbor interpolation using PyTorch as reference
    """
    # Extract spatial dimensions (H, W)
    target_size = output_shape[2:]  # Skip batch and channel dimensions
    
    # Use PyTorch's interpolate function with nearest mode
    if input_tensor.dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
        # 对于整数类型，先转换为 float32，进行插值，再转换回原类型
        original_dtype = input_tensor.dtype
        
        # 转换为 float32 进行插值
        float_input = input_tensor.float()
        result = F.interpolate(
            float_input, 
            size=target_size, 
            mode='nearest'
        )
        
        # 转换回原始类型
        result = result.to(original_dtype)
    else:
        result = F.interpolate(
            input_tensor, 
            size=target_size, 
            mode='nearest'
        )
    
    output_tensor.copy_(result)


def test(
    handle,
    device,
    input_shape,
    input_stride,
    output_shape,
    output_stride,
    tensor_dtype=InfiniDtype.F16,
    sync=None,
):
    # Create input and output tensors
    input_tensor = TestTensor(input_shape, input_stride, dt=tensor_dtype, device=device, scale=1.0)
    output_tensor = TestTensor(output_shape, output_stride, dt=tensor_dtype, device=device)

    print(
        f"Testing InterpolateNearest on {InfiniDeviceNames[device]} with "
        f"input_shape: {input_shape}, output_shape: {output_shape}, "
        f"input_stride: {input_stride}, output_stride: {output_stride}, "
        f"dtype: {InfiniDtypeNames[tensor_dtype]}"
    )

    # Compute reference result using PyTorch
    interpolate_nearest(
        input_tensor.torch_tensor(),
        output_shape,
        output_tensor.torch_tensor()
    )

    if sync is not None:
        sync()

    # Create descriptor for our interpolate_nearest operator
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateInterpolateNearestDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, output_tensor]:
        if tensor is not None:
            tensor.destroy_desc()

    # Get workspace size
    workspace_size = ctypes.c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetInterpolateNearestWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output_tensor.device)

    def lib_interpolate_nearest():
        check_error(
            LIBINFINIOP.infiniopInterpolateNearest(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output_tensor.data(),
                input_tensor.data(),
                None,
            )
        )
    
    # Execute the operation
    lib_interpolate_nearest()
    
    # Check results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(output_tensor.actual_tensor(), output_tensor.torch_tensor(), atol=atol, rtol=rtol)
    
    assert torch.allclose(
        output_tensor.actual_tensor(), 
        output_tensor.torch_tensor(), 
        atol=atol, 
        rtol=rtol
    ), f"Results don't match for shape {input_shape} -> {output_shape}"

    # Profiling workflow
    if PROFILE:
        profile_operation(
            "PyTorch", 
            lambda: interpolate_nearest(
                input_tensor.torch_tensor(), 
                output_shape, 
                output_tensor.torch_tensor()
            ), 
            device, 
            NUM_PRERUN, 
            NUM_ITERATIONS
        )
        profile_operation(
            "    lib", 
            lambda: lib_interpolate_nearest(), 
            device, 
            NUM_PRERUN, 
            NUM_ITERATIONS
        )

    # Clean up
    check_error(LIBINFINIOP.infiniopDestroyInterpolateNearestDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
