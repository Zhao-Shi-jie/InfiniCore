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
    (8, [7, 1, 3, 0]),
]

_TENSOR_DTYPES = [InfiniDtype.F32]
_INDEX_DTYPES = [InfiniDtype.I32]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
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
        f"Testing SparseGather on {InfiniDeviceNames[device]} with shape:({size},), "
        f"nnz:{len(indices)}, dtype:{InfiniDtypeNames[dtype]}, "
        f"index_dtype:{InfiniDtypeNames[index_dtype]}"
    )

    nnz = len(indices)
    indices_tensor = TestTensor.from_torch(torch.tensor(indices), index_dtype, device)
    values = TestTensor((nnz,), None, dtype, device)
    x = TestTensor((size,), None, dtype, device)
    out = TestTensor((nnz,), None, dtype, device, mode="zeros")
    ans = x.torch_tensor()[torch.tensor(indices, dtype=torch.int64, device=x.torch_tensor().device)]

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
        LIBINFINIOP.infiniopCreateSparseGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            spvec_desc,
            x.descriptor,
            out.data(),
            x.data(),
        )
    )

    for tensor in [indices_tensor, values, x, out]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSparseGatherWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopSparseGather(
            descriptor,
            workspace.data(),
            workspace_size.value,
            out.data(),
            x.data(),
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    check_error(LIBINFINIOP.infiniopDestroySparseGatherDescriptor(descriptor))
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
