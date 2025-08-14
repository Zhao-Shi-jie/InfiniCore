import torch
import ctypes
from ctypes import c_uint64, c_bool

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
from typing import Tuple
import math
from torch.nn import functional as F

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

_TEST_CASES = [
    ((1, 3, 8), None, (2,), (2,), (0,), False),
    ((2, 4, 16), None, (3,), (2,), (1,), False),
    ((1, 1, 4, 4), None, (2, 2), (2, 2), (0, 0), False),
    ((2, 3, 8, 8), None, (3, 3), (2, 2), (1, 1), False),
    ((1, 64, 32, 32), None, (2, 2), (2, 2), (0, 0), False),
    ((4, 128, 16, 16), None, (3, 3), (1, 1), (1, 1), False),
    ((1, 1, 4, 4, 4), None, (2, 2, 2), (2, 2, 2), (0, 0, 0), False),
    ((2, 2, 8, 8, 8), None, (2, 3, 3), (2, 2, 2), (0, 1, 1), False),
    ((1, 1, 7, 7), None, (3, 3), (2, 2), (1, 1), True),
    ((1, 2, 5), None, (3,), (2,), (0,), True),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

def avg_pool(input_tensor, kernel_size, stride, padding, ceil_mode, output_tensor):
    ndim = len(input_tensor.shape) - 2
    if ndim == 1:
        result = F.avg_pool1d(input_tensor, kernel_size[0], stride[0], padding[0], ceil_mode=ceil_mode)
    elif ndim == 2:
        result = F.avg_pool2d(input_tensor, kernel_size, stride, padding, ceil_mode=ceil_mode)
    elif ndim == 3:
        result = F.avg_pool3d(input_tensor, kernel_size, stride, padding, ceil_mode=ceil_mode)
    else:
        raise ValueError(f"Unsupported spatial dimensions: {ndim}")
    output_tensor.copy_(result)

def infer_output_shape(input_shape, kernel_size, stride, padding, ceil_mode):
    def calc_output_size(input_size, k, s, p, ceil_mode):
        return math.ceil((input_size + 2 * p - k) / s + 1) if ceil_mode else math.floor((input_size + 2 * p - k) / s + 1)
    batch, channel, *spatial = input_shape
    output_spatial = [calc_output_size(spatial[i], kernel_size[i], stride[i], padding[i], ceil_mode) for i in range(len(spatial))]
    return (batch, channel) + tuple(output_spatial)

def tuple_to_void_p(py_tuple: Tuple):
    arr = (ctypes.c_uint64 * len(py_tuple))(*py_tuple)
    return ctypes.cast(arr, ctypes.c_void_p)

def test(handle, device, input_shape, input_stride, kernel_size, stride, padding, ceil_mode, tensor_dtype=InfiniDtype.F16, sync=None):
    input_tensor = TestTensor(input_shape, input_stride, dt=tensor_dtype, device=device, scale=1.0)
    output_shape = infer_output_shape(input_shape, kernel_size, stride, padding, ceil_mode)
    output_tensor = TestTensor(output_shape, None, dt=tensor_dtype, device=device)

    print(f"Testing AvgPool on {InfiniDeviceNames[device]} with input_shape: {input_shape}, kernel_size: {kernel_size}, stride: {stride}, padding: {padding}, ceil_mode: {ceil_mode}, dtype: {InfiniDtypeNames[tensor_dtype]}")

    avg_pool(input_tensor.torch_tensor(), kernel_size, stride, padding, ceil_mode, output_tensor.torch_tensor())

    if sync: sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateAvgPoolDescriptor(
        handle,
        ctypes.byref(descriptor),
        output_tensor.descriptor,
        input_tensor.descriptor,
        tuple_to_void_p(kernel_size),
        tuple_to_void_p(stride),
        tuple_to_void_p(padding),
        c_bool(ceil_mode),
    ))

    for tensor in [input_tensor, output_tensor]:
        if tensor: tensor.destroy_desc()

    workspace_size = ctypes.c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetAvgPoolWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = TestWorkspace(workspace_size.value, output_tensor.device)

    def lib_avg_pool():
        check_error(LIBINFINIOP.infiniopAvgPool(
            descriptor,
            workspace.data(),
            workspace_size.value,
            output_tensor.data(),
            input_tensor.data(),
            None,
        ))

    lib_avg_pool()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(output_tensor.actual_tensor(), output_tensor.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(output_tensor.actual_tensor(), output_tensor.torch_tensor(), atol=atol, rtol=rtol), f"Mismatch for shape {input_shape}, kernel {kernel_size}"

    if PROFILE:
        profile_operation("PyTorch", lambda: avg_pool(input_tensor.torch_tensor(), kernel_size, stride, padding, ceil_mode, output_tensor.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_avg_pool, device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyAvgPoolDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
