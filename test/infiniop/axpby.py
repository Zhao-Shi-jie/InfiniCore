import ctypes
import random
from ctypes import c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    infiniopSpVecDescriptor_t,
    test_operator,
)

_INDEX_DTYPES = [InfiniDtype.I32, InfiniDtype.I64]
_TENSOR_DTYPES = [InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False


def _generate_cases():
    random.seed(42)
    configs = [
        (256, 0.03, 1.0, 0.0),
        (4096, 0.01, 0.5, 1.0),
        (10000, 0.002, -1.25, 0.25),
    ]
    cases = []
    for size, density, alpha, beta in configs:
        nnz = max(1, int(size * density))
        indices = sorted(random.sample(range(size), nnz))
        cases.append((size, density, indices, alpha, beta))
    return cases


_TEST_CASES = _generate_cases()


def test(
    handle,
    device,
    size,
    density,
    indices,
    alpha,
    beta,
    index_dtype=InfiniDtype.I32,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing Axpby on {InfiniDeviceNames[device]} with size:{size}, density:{density:.6f}, "
        f"alpha:{alpha}, beta:{beta}, dtype:{InfiniDtypeNames[dtype]}, "
        f"index_dtype:{InfiniDtypeNames[index_dtype]}"
    )

    nnz = len(indices)
    indices_tensor = TestTensor.from_torch(torch.tensor(indices), index_dtype, device)
    x_values = TestTensor((nnz,), None, dtype, device)
    y = TestTensor((size,), None, dtype, device)

    ans = beta * y.torch_tensor().clone()
    ans[indices_tensor.torch_tensor().long()] += alpha * x_values.torch_tensor()

    if sync is not None:
        sync()

    spvec_desc = infiniopSpVecDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSpVecDescriptor(
            ctypes.byref(spvec_desc),
            size,
            nnz,
            x_values.descriptor,
            indices_tensor.descriptor,
            x_values.data(),
            indices_tensor.data(),
        )
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAxpbyDescriptor(
            handle,
            ctypes.byref(descriptor),
            spvec_desc,
            y.descriptor,
        )
    )

    for tensor in [x_values, indices_tensor, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAxpbyWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopAxpby(
            descriptor,
            workspace.data(),
            workspace_size.value,
            y.data(),
            alpha,
            beta,
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    check_error(LIBINFINIOP.infiniopDestroyAxpbyDescriptor(descriptor))
    check_error(LIBINFINIOP.infiniopDestroySpVecDescriptor(spvec_desc))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug

    for device in get_test_devices(args):
        test_cases = [
            (*case, index_dtype)
            for case in _TEST_CASES
            for index_dtype in _INDEX_DTYPES
        ]
        test_operator(device, test, test_cases, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
