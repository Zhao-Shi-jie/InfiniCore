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
    infiniopSpMatDescriptor_t,
    test_operator,
)


def _generate_sddmm_cases():
    cases = []
    random.seed(42)
    # (rows, cols, k, density, alpha, beta)
    configs = [
        (128, 128, 128, 0.01, 1.0, 0.0),
        (1024, 1024, 1024, 0.01, 0.5, 1.0),
        (4096, 4096, 4096, 0.01, -1.25, 0.25),
    ]
    for rows, cols, k, density, alpha, beta in configs:
        total = rows * cols
        nnz = min(total, max(1, int(round(total * density))))
        positions = sorted(random.sample(range(total), nnz))
        crow = [0] * (rows + 1)
        col = []
        for pos in positions:
            row = pos // cols
            col.append(pos % cols)
            crow[row + 1] += 1
        for row in range(rows):
            crow[row + 1] += crow[row]
        cases.append((alpha, beta, rows, cols, k, crow, col))
    return cases


_BASE_TEST_CASES = _generate_sddmm_cases()

_TENSOR_DTYPES = [InfiniDtype.F32]
_INDEX_DTYPES = [InfiniDtype.I32, InfiniDtype.I64]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False


def sampled_mm(a, b, values, rows, crow, col, alpha, beta):
    mm = torch.matmul(a, b)
    result = values.clone()
    for row in range(rows):
        for ptr in range(crow[row], crow[row + 1]):
            result[ptr] = alpha * mm[row, col[ptr]] + beta * values[ptr]
    return result


def test(
    handle,
    device,
    alpha,
    beta,
    rows,
    cols,
    k,
    crow,
    col,
    index_dtype=InfiniDtype.I32,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing SDDMM on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
        f" shape:({rows}, {k}) x ({k}, {cols}), dtype:{InfiniDtypeNames[dtype]},"
        f" index_dtype:{InfiniDtypeNames[index_dtype]}"
    )

    nnz = len(col)
    crow_tensor = TestTensor.from_torch(torch.tensor(crow), index_dtype, device)
    col_tensor = TestTensor.from_torch(torch.tensor(col), index_dtype, device)
    values = TestTensor((nnz,), None, dtype, device)
    a = TestTensor((rows, k), None, dtype, device)
    b = TestTensor((k, cols), None, dtype, device)
    ans = sampled_mm(
        a.torch_tensor(),
        b.torch_tensor(),
        values.torch_tensor(),
        rows,
        crow,
        col,
        alpha,
        beta,
    )

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
        LIBINFINIOP.infiniopCreateSDDMMDescriptor(
            handle,
            ctypes.byref(descriptor),
            spmat_desc,
            a.descriptor,
            b.descriptor,
        )
    )

    for tensor in [values, crow_tensor, col_tensor, a, b]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSDDMMWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopSDDMM(
            descriptor,
            workspace.data(),
            workspace_size.value,
            values.data(),
            a.data(),
            b.data(),
            alpha,
            beta,
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(values.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(values.actual_tensor(), ans, atol=atol, rtol=rtol)

    check_error(LIBINFINIOP.infiniopDestroySDDMMDescriptor(descriptor))
    check_error(LIBINFINIOP.infiniopDestroySpMatDescriptor(spmat_desc))


if __name__ == "__main__":
    torch.manual_seed(42)
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
