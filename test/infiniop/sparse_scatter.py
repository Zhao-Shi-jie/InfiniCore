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
    # size, density
    (6, 0.5),
    (128, 0.04),
    (1024, 0.02),
    (4096, 0.01),
]

_TENSOR_DTYPES = [InfiniDtype.F32]
_INDEX_DTYPES = [
    InfiniDtype.I32,
    # InfiniDtype.I64,
]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False
_RANDOM_SEED = 42


def generate_indices(size, density):
    nnz = min(size, max(1, int(round(size * density))))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_RANDOM_SEED + size)
    indices = torch.randperm(size, generator=generator)[:nnz]
    indices, _ = torch.sort(indices)
    return indices.tolist()


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
        f"Testing SparseScatter on {InfiniDeviceNames[device]} with shape:({size},), "
        f"nnz:{len(indices)}, dtype:{InfiniDtypeNames[dtype]}, "
        f"index_dtype:{InfiniDtypeNames[index_dtype]}"
    )

    nnz = len(indices)
    indices_tensor = TestTensor.from_torch(torch.tensor(indices), index_dtype, device)
    values = TestTensor((nnz,), None, dtype, device)
    out = TestTensor((size,), None, dtype, device, mode="zeros")
    ans = torch.zeros_like(out.torch_tensor())
    ans[torch.tensor(indices, dtype=torch.int64, device=ans.device)] = values.torch_tensor()

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
        LIBINFINIOP.infiniopCreateSparseScatterDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            spvec_desc,
        )
    )

    for tensor in [indices_tensor, values, out]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSparseScatterWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopSparseScatter(
            descriptor,
            workspace.data(),
            workspace_size.value,
            out.data(),
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    check_error(LIBINFINIOP.infiniopDestroySparseScatterDescriptor(descriptor))
    check_error(LIBINFINIOP.infiniopDestroySpVecDescriptor(spvec_desc))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug

    for device in get_test_devices(args):
        test_cases = [
            (size, generate_indices(size, density), index_dtype)
            for size, density in _BASE_TEST_CASES
            for index_dtype in _INDEX_DTYPES
        ]
        test_operator(device, test, test_cases, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
