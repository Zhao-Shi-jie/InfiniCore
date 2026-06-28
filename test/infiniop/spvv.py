import ctypes
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

_BASE_TEST_CASES = [
    # size, indices
    (6, [0, 2, 5]),
    (8, [1, 3, 4, 7]),
]

_TENSOR_DTYPES = [
    # InfiniDtype.F16,
    # InfiniDtype.BF16,
    InfiniDtype.F32,
]
_INDEX_DTYPES = [InfiniDtype.I32, InfiniDtype.I64]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False


def test(
    handle,
    device,
    size,
    indices,
    index_dtype=InfiniDtype.I32,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing SpVV on {InfiniDeviceNames[device]} with shape:({size},) dot ({size},), dtype:{InfiniDtypeNames[dtype]},"
        f" index_dtype:{InfiniDtypeNames[index_dtype]}"
    )

    nnz = len(indices)
    indices_tensor = TestTensor.from_torch(torch.tensor(indices), index_dtype, device)
    values = TestTensor((nnz,), None, dtype, device)
    x = TestTensor((size,), None, dtype, device)
    y = TestTensor((), None, dtype, device, mode="ones")
    ans = TestTensor((), None, dtype, device, mode="zeros")

    sparse_dense = torch.zeros(size, dtype=values.torch_tensor().dtype, device=values.torch_tensor().device)
    sparse_dense[indices_tensor.torch_tensor().long()] = values.torch_tensor()
    ans.set_tensor(torch.dot(sparse_dense, x.torch_tensor()))

    if sync is not None:
        sync()

    spvec_desc = infiniopSpVecDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSpVecDescriptor(
            ctypes.byref(spvec_desc),
            size,
            nnz,
            values.descriptor,
            indices_tensor.descriptor,
            values.data(),
            indices_tensor.data(),
        )
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSpVVDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            spvec_desc,
            x.descriptor,
            x.data(),
        )
    )

    for tensor in [values, indices_tensor, x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSpVVWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopSpVV(
            descriptor,
            workspace.data(),
            workspace_size.value,
            y.data(),
            x.data(),
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    check_error(LIBINFINIOP.infiniopDestroySpVVDescriptor(descriptor))
    check_error(LIBINFINIOP.infiniopDestroySpVecDescriptor(spvec_desc))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug

    for device in get_test_devices(args):
        test_cases = [
            (*case, index_dtype)
            for case in _BASE_TEST_CASES
            for index_dtype in _INDEX_DTYPES
        ]
        test_operator(device, test, test_cases, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
