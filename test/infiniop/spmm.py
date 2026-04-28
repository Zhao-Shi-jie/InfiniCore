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
    infiniopSpMatDescriptor_t,
    test_operator,
)

_BASE_TEST_CASES = [
    # alpha, beta, rows, cols, n, crow, col
    (1.0, 0.0, 3, 4, 2, [0, 2, 3, 5], [0, 2, 1, 0, 3]),
    (0.5, 1.0, 4, 5, 3, [0, 1, 1, 3, 4], [2, 0, 4, 1]),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]
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
    alpha,
    beta,
    rows,
    cols,
    n,
    crow,
    col,
    index_dtype=InfiniDtype.I32,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing SpMM on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
        f" shape:({rows}, {cols}) x ({cols}, {n}), dtype:{InfiniDtypeNames[dtype]},"
        f" index_dtype:{InfiniDtypeNames[index_dtype]}"
    )

    nnz = len(col)
    crow_tensor = TestTensor.from_torch(torch.tensor(crow), index_dtype, device)
    col_tensor = TestTensor.from_torch(torch.tensor(col), index_dtype, device)
    values = TestTensor((nnz,), None, dtype, device)
    b = TestTensor((cols, n), None, dtype, device)
    c = TestTensor((rows, n), None, dtype, device, mode="ones")
    ans = TestTensor((rows, n), None, dtype, device, mode="zeros")

    sparse = torch.sparse_csr_tensor(
        crow_tensor.torch_tensor(),
        col_tensor.torch_tensor(),
        values.torch_tensor(),
        size=(rows, cols),
        device=values.torch_tensor().device,
    )
    ans.update_torch_tensor(alpha * torch.matmul(sparse, b.torch_tensor()) + beta * c.torch_tensor())

    if sync is not None:
        sync()

    spmat_desc = infiniopSpMatDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCsrSpMatDescriptor(
            ctypes.byref(spmat_desc),
            rows,
            cols,
            nnz,
            values.descriptor,
            crow_tensor.descriptor,
            col_tensor.descriptor,
            values.data(),
            crow_tensor.data(),
            col_tensor.data(),
        )
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSpMMDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            spmat_desc,
            b.descriptor,
        )
    )

    for tensor in [values, crow_tensor, col_tensor, b, c]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSpMMWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopSpMM(
            descriptor,
            workspace.data(),
            workspace_size.value,
            c.data(),
            b.data(),
            alpha,
            beta,
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    check_error(LIBINFINIOP.infiniopDestroySpMMDescriptor(descriptor))
    check_error(LIBINFINIOP.infiniopDestroySpMatDescriptor(spmat_desc))


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
