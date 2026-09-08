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

_CSR_PIPELINE_ROWS = 128
_CSR_PIPELINE_CROW = [2 * ((row + 1) // 2) for row in range(_CSR_PIPELINE_ROWS + 1)]
_CSR_PIPELINE_COL = [
    col for _ in range((_CSR_PIPELINE_ROWS + 1) // 2) for col in (0, 2)
]
_CSR_PIPELINE_TEST_CASES = [
    # Alternating nonempty/empty rows; wide and tall enough to reuse accumulators per task.
    (
        0.75,
        0.0,
        _CSR_PIPELINE_ROWS,
        3,
        25000,
        _CSR_PIPELINE_CROW,
        _CSR_PIPELINE_COL,
    ),
    (
        0.75,
        0.5,
        _CSR_PIPELINE_ROWS,
        3,
        25000,
        _CSR_PIPELINE_CROW,
        _CSR_PIPELINE_COL,
    ),
]

_TENSOR_DTYPES = [
    # InfiniDtype.F16,
    # InfiniDtype.BF16,
    InfiniDtype.F32
]
_INDEX_DTYPES = [InfiniDtype.I32, InfiniDtype.I64]
_SPARSE_FORMATS = ["csr", "ell", "sell", "sell_sigma_c"]
_SELL_SLICE_HEIGHT = 2
_SELL_SIGMA = 4

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False


def csr_to_dense(values, rows, cols, crow, col):
    device = values.device
    crow_tensor = torch.tensor(crow, dtype=torch.int64, device=device)
    col_tensor = torch.tensor(col, dtype=torch.int64, device=device)
    row_counts = crow_tensor[1:] - crow_tensor[:-1]
    row_tensor = torch.repeat_interleave(
        torch.arange(rows, dtype=torch.int64, device=device), row_counts
    )
    dense = torch.zeros((rows, cols), dtype=values.dtype, device=device)
    dense.index_put_((row_tensor, col_tensor), values, accumulate=True)
    return dense


def csr_row_lengths(rows, crow):
    return [crow[row + 1] - crow[row] for row in range(rows)]


def csr_to_ell(values, rows, crow, col):
    row_lengths = csr_row_lengths(rows, crow)
    ell_width = max(row_lengths) if row_lengths else 0
    ell_values = torch.zeros(
        (rows, ell_width), dtype=values.dtype, device=values.device
    )
    ell_col = torch.zeros((rows, ell_width), dtype=torch.int64, device=values.device)
    for row in range(rows):
        for slot, ptr in enumerate(range(crow[row], crow[row + 1])):
            ell_values[row, slot] = values[ptr]
            ell_col[row, slot] = col[ptr]
    return ell_values, ell_col, ell_width


def sigma_sorted_rows(rows, crow, sigma):
    lengths = csr_row_lengths(rows, crow)
    order = []
    for begin in range(0, rows, sigma):
        window = list(range(begin, min(begin + sigma, rows)))
        window.sort(key=lambda row: lengths[row], reverse=True)
        order.extend(window)
    return order


def csr_to_sell(values, rows, crow, col, slice_height, row_order=None):
    if row_order is None:
        row_order = list(range(rows))

    row_lengths = csr_row_lengths(rows, crow)
    num_slices = (rows + slice_height - 1) // slice_height
    slice_offsets = [0]
    sell_values = []
    sell_col = []

    for slice_id in range(num_slices):
        begin = slice_id * slice_height
        storage_rows = row_order[begin : begin + slice_height]
        slice_width = max((row_lengths[row] for row in storage_rows), default=0)
        for slot in range(slice_width):
            for row_in_slice in range(slice_height):
                if row_in_slice >= len(storage_rows):
                    sell_values.append(
                        torch.zeros((), dtype=values.dtype, device=values.device)
                    )
                    sell_col.append(0)
                    continue
                row = storage_rows[row_in_slice]
                row_nnz = row_lengths[row]
                if slot < row_nnz:
                    ptr = crow[row] + slot
                    sell_values.append(values[ptr])
                    sell_col.append(col[ptr])
                else:
                    sell_values.append(
                        torch.zeros((), dtype=values.dtype, device=values.device)
                    )
                    sell_col.append(0)
        slice_offsets.append(len(sell_values))

    if sell_values:
        values_tensor = torch.stack(sell_values).to(dtype=values.dtype)
    else:
        values_tensor = torch.empty((0,), dtype=values.dtype, device=values.device)
    col_tensor = torch.tensor(sell_col, dtype=torch.int64, device=values.device)
    offsets_tensor = torch.tensor(
        slice_offsets, dtype=torch.int64, device=values.device
    )
    row_indices_tensor = torch.tensor(
        row_order, dtype=torch.int64, device=values.device
    )
    return values_tensor, offsets_tensor, col_tensor, row_indices_tensor, num_slices


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
    sparse_format="csr",
    index_dtype=InfiniDtype.I32,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing SpMM on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
        f" shape:({rows}, {cols}) x ({cols}, {n}), dtype:{InfiniDtypeNames[dtype]},"
        f" index_dtype:{InfiniDtypeNames[index_dtype]}, format:{sparse_format}"
    )

    nnz = len(col)
    crow_tensor = TestTensor.from_torch(torch.tensor(crow), index_dtype, device)
    col_tensor = TestTensor.from_torch(torch.tensor(col), index_dtype, device)
    values = TestTensor((nnz,), None, dtype, device)
    b = TestTensor((cols, n), None, dtype, device)
    c = TestTensor((rows, n), None, dtype, device, mode="ones")
    ans = TestTensor((rows, n), None, dtype, device, mode="zeros")

    sparse = csr_to_dense(values.torch_tensor(), rows, cols, crow, col)
    ans.set_tensor(
        alpha * torch.matmul(sparse, b.torch_tensor()) + beta * c.torch_tensor()
    )

    if sync is not None:
        sync()

    spmat_desc = infiniopSpMatDescriptor_t()
    spmat_tensors = [values, crow_tensor, col_tensor]
    if sparse_format == "csr":
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
    elif sparse_format == "ell":
        ell_values, ell_col, ell_width = csr_to_ell(
            values.torch_tensor(), rows, crow, col
        )
        ell_values_tensor = TestTensor.from_torch(ell_values, dtype, device)
        ell_col_tensor = TestTensor.from_torch(ell_col, index_dtype, device)
        spmat_tensors += [ell_values_tensor, ell_col_tensor]
        check_error(
            LIBINFINIOP.infiniopCreateEllSpMatDescriptor(
                ctypes.byref(spmat_desc),
                rows,
                cols,
                nnz,
                ell_width,
                ell_values_tensor.descriptor,
                ell_col_tensor.descriptor,
                ell_values_tensor.data(),
                ell_col_tensor.data(),
            )
        )
    elif sparse_format == "sell":
        sell_values, slice_offsets, sell_col, _, num_slices = csr_to_sell(
            values.torch_tensor(), rows, crow, col, _SELL_SLICE_HEIGHT
        )
        sell_values_tensor = TestTensor.from_torch(sell_values, dtype, device)
        slice_offsets_tensor = TestTensor.from_torch(slice_offsets, index_dtype, device)
        sell_col_tensor = TestTensor.from_torch(sell_col, index_dtype, device)
        spmat_tensors += [sell_values_tensor, slice_offsets_tensor, sell_col_tensor]
        check_error(
            LIBINFINIOP.infiniopCreateSellSpMatDescriptor(
                ctypes.byref(spmat_desc),
                rows,
                cols,
                nnz,
                _SELL_SLICE_HEIGHT,
                num_slices,
                sell_values_tensor.descriptor,
                slice_offsets_tensor.descriptor,
                sell_col_tensor.descriptor,
                sell_values_tensor.data(),
                slice_offsets_tensor.data(),
                sell_col_tensor.data(),
            )
        )
    elif sparse_format == "sell_sigma_c":
        row_order = sigma_sorted_rows(rows, crow, _SELL_SIGMA)
        sell_values, slice_offsets, sell_col, row_indices, num_slices = csr_to_sell(
            values.torch_tensor(), rows, crow, col, _SELL_SLICE_HEIGHT, row_order
        )
        sell_values_tensor = TestTensor.from_torch(sell_values, dtype, device)
        slice_offsets_tensor = TestTensor.from_torch(slice_offsets, index_dtype, device)
        sell_col_tensor = TestTensor.from_torch(sell_col, index_dtype, device)
        row_indices_tensor = TestTensor.from_torch(row_indices, index_dtype, device)
        spmat_tensors += [
            sell_values_tensor,
            slice_offsets_tensor,
            sell_col_tensor,
            row_indices_tensor,
        ]
        check_error(
            LIBINFINIOP.infiniopCreateSellSigmaCSpMatDescriptor(
                ctypes.byref(spmat_desc),
                rows,
                cols,
                nnz,
                _SELL_SLICE_HEIGHT,
                _SELL_SIGMA,
                num_slices,
                sell_values_tensor.descriptor,
                slice_offsets_tensor.descriptor,
                sell_col_tensor.descriptor,
                row_indices_tensor.descriptor,
                sell_values_tensor.data(),
                slice_offsets_tensor.data(),
                sell_col_tensor.data(),
                row_indices_tensor.data(),
            )
        )
    else:
        raise ValueError(f"Unsupported sparse format: {sparse_format}")

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

    for tensor in spmat_tensors + [b, c]:
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
            (*case, sparse_format, index_dtype)
            for case in _BASE_TEST_CASES
            for sparse_format in _SPARSE_FORMATS
            for index_dtype in _INDEX_DTYPES
        ] + [
            (*case, "csr", index_dtype)
            for case in _CSR_PIPELINE_TEST_CASES
            for index_dtype in _INDEX_DTYPES
        ]
        test_operator(device, test, test_cases, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
