import torch
import ctypes
from ctypes import c_uint64
import numpy as np

from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    TestWorkspace,
    InfiniDeviceEnum,
)
from torch.nn import functional as F

_TEST_CASES = [
    ((4, 100),),           # 标准 2D 分类，(N, C)
    ((2, 10, 32, 32),),    # 4D 图像分类，(N, C, H, W)
    ((1, 5, 4, 4),),       # 小 batch 小图，低精度验证
    ((3, 7, 8),),          # 3D logits，模拟 (N, C, L)
    ((2, 6, 2, 3, 3),),    # 5D logits，模拟 3D 卷积输出
]

_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}
DEBUG = False
PROFILE = False

def cross_entropy_loss_pytorch(logits, target, loss_tensor):
    loss_ref = F.cross_entropy(logits, target.long(), reduction="none")
    loss_tensor.copy_(loss_ref)

def get_strides(shape):
    if not shape: return tuple()
    strides = [1]
    for s in reversed(shape[1:]):
        strides.insert(0, strides[0] * s)
    return tuple(strides)

def test(
    handle, device, logits_shape,
    tensor_dtype=InfiniDtype.F32, sync=None,
):
    target_shape = (logits_shape[0],) + logits_shape[2:]
    loss_shape = target_shape

    print(f"Testing CrossEntropyLoss on {InfiniDeviceNames[device]} with logits_shape: {logits_shape}, dtype:{InfiniDtypeNames[tensor_dtype]}")

    logits = TestTensor(logits_shape, get_strides(logits_shape), dt=tensor_dtype, device=device)
    loss = TestTensor(loss_shape, get_strides(loss_shape), dt=tensor_dtype, device=device)
    
    num_classes = logits_shape[1]
    target_torch = torch.randint(0, num_classes, target_shape, dtype=torch.int, device=logits.torch_tensor().device)
    target = TestTensor.from_torch(target_torch, dt=InfiniDtype.I32, device=device)

    cross_entropy_loss_pytorch(logits.torch_tensor(), target.torch_tensor(), loss.torch_tensor())
    if sync: sync()

    logits_torch = logits.torch_tensor()
    target_torch = target.torch_tensor()
    softmax_probs = torch.softmax(logits_torch, dim=1)
    
    # 展平 logits 和 target，以逐位置检查
    # (N, C, H, W) → N * H * W 个位置，每个位置有 C 个 logit
    N = logits_shape[0]
    C = logits_shape[1]
    spatial_size = int(torch.tensor(logits_shape[2:]).prod().item()) if len(logits_shape) > 2 else 1

    for i in range(min(10, N * spatial_size)):
        n = i // spatial_size
        s = i % spatial_size
        if len(logits_shape) == 2:
            # (N, C)
            t = target_torch[n]
            prob = softmax_probs[n, t]
        elif len(logits_shape) >= 3:
            # 通用高维结构：(N, C, D1, D2, ..., Dn)
            spatial_indices = []
            remaining = s
            for dim in reversed(logits_shape[2:]):
                spatial_indices.insert(0, remaining % dim)
                remaining //= dim
            t = target_torch[(n, *spatial_indices)]
            prob = softmax_probs[(n, t.item(), *spatial_indices)]

        
        loss_val = -torch.log(prob)
        # print(f"[DEBUG] idx={i} target={t.item()} prob={prob.item():.6f} loss={loss_val.item():.6f}")


    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCrossEntropyLossDescriptor(
            handle, ctypes.byref(descriptor),
            loss.descriptor, logits.descriptor, target.descriptor,
        )
    )
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCrossEntropyLossWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)
    check_error(
        LIBINFINIOP.infiniopCrossEntropyLoss(
            descriptor, workspace.data(), workspace_size.value,
            loss.data(), logits.data(), target.data(), None,
        )
    )

    if sync: sync()
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    actual_flat = loss.actual_tensor().flatten()
    torch_flat = loss.torch_tensor().flatten()

    if not torch.allclose(actual_flat, torch_flat, atol=atol, rtol=rtol):
        print("--- ERROR ANALYSIS ---")
        print("Target (flattened):", target.torch_tensor().flatten()[:10])
        print("实际输出 (flattened):", actual_flat[:10])
        print("参考输出 (flattened):", torch_flat[:10])
    
    assert torch.allclose(actual_flat, torch_flat, atol=atol, rtol=rtol)
    check_error(LIBINFINIOP.infiniopDestroyCrossEntropyLossDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mAll CrossEntropyLoss tests passed!\033[0m")