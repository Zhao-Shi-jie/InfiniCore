# Sparse Support Guide

This document describes the sparse matrix infrastructure in InfiniCore and the expected workflow for adding sparse operators. The current sparse support is intentionally incremental: dense `Tensor` remains unchanged, and sparse formats are represented by descriptors that reference existing dense tensors.

## Architecture Overview

Sparse support is split across two layers:

- **InfiniOP** provides the C ABI, descriptors, backend dispatch, and device kernels.
- **InfiniCore** wraps InfiniOP descriptors with C++ objects, graph operators, pybind11 bindings, and Python APIs.

Sparse matrices are not passed to operators as three unrelated tensors. Instead, callers build a `SpMat` object first, then pass that object to sparse operators. This mirrors cuSPARSE-style usage while keeping the underlying storage based on existing dense `Tensor` objects.

## Sparse Matrix Infrastructure

The current supported sparse format is CSR.

InfiniOP exposes the low-level sparse descriptor in:

- `include/infiniop/spmat_descriptor.h`
- `src/infiniop/spmat.h`
- `src/infiniop/spmat_descriptor.cc`

`infiniopCreateCsrSpMatDescriptor` records matrix shape, `nnz`, tensor descriptors, and raw pointers for:

- `values`: nonzero values, same dtype as dense compute tensors.
- `crow_indices`: CSR row offsets, dtype `I32` or `I64`.
- `col_indices`: CSR column indices, dtype `I32` or `I64`.

InfiniCore exposes the C++ wrapper in:

- `include/infinicore/spmat.hpp`
- `src/infinicore/spmat.cc`

Python exposes:

- `python/infinicore/spmat.py`
- `infinicore.csr_spmat(crow_indices, col_indices, values, size)`
- `infinicore.SpMat`

Example:

```python
crow = infinicore.from_list([0, 2, 3], dtype=infinicore.int64, device=device)
col = infinicore.from_list([0, 2, 1], dtype=infinicore.int64, device=device)
values = infinicore.from_list([1.0, 2.0, 3.0], dtype=infinicore.float32, device=device)
a = infinicore.csr_spmat(crow, col, values, (2, 3))
```

## Existing Sparse Operator: SpMM

SpMM computes:

```text
C = alpha * A_sparse @ B_dense + beta * C
```

Implemented files:

- InfiniOP public API: `include/infiniop/ops/spmm.h`
- InfiniOP operator core: `src/infiniop/ops/spmm/`
- CPU backend: `src/infiniop/ops/spmm/cpu/`
- NVIDIA backend: `src/infiniop/ops/spmm/nvidia/`
- InfiniCore C++ API: `include/infinicore/ops/spmm.hpp`
- InfiniCore operator wrapper: `src/infinicore/ops/spmm/`
- pybind11 binding: `src/infinicore/pybind11/ops/spmm.hpp`
- Python API: `python/infinicore/ops/spmm.py`
- Tests: `test/infiniop/spmm.py`, `test/infinicore/ops/spmm.py`

Use it from Python:

```python
c = infinicore.spmm(a, b)
infinicore.spmm(a, b, out=c)
c = a @ b
```

## Adding a Sparse Operator

Follow the same shape as SpMM.

1. Add the InfiniOP public header under `include/infiniop/ops/<op>.h`, then include it from `include/infiniop.h`.
2. Add operator metadata and validation under `src/infiniop/ops/<op>/info.h`. Validate sparse format, dense tensor ranks, shapes, strides, dtype compatibility, and index dtype.
3. Add a descriptor declaration header under `src/infiniop/ops/<op>/<op>.h`. If the descriptor stores sparse matrix descriptors or raw sparse pointers, be careful with lifetime and cache behavior.
4. Implement backend dispatch in `src/infiniop/ops/<op>/operator.cc`.
5. Implement device backends under `src/infiniop/ops/<op>/cpu/`, `nvidia/`, etc.
6. Register ctypes bindings in `test/infiniop/libinfiniop/op_register.py` and add an InfiniOP test in `test/infiniop/<op>.py`.
7. Add InfiniCore C++ API in `include/infinicore/ops/<op>.hpp` and include it from `include/infinicore/ops.hpp`.
8. Add InfiniCore graph/operator wrapper in `src/infinicore/ops/<op>/`.
9. Add pybind11 binding in `src/infinicore/pybind11/ops/<op>.hpp` and call it from `src/infinicore/pybind11/ops.hpp`.
10. Add Python API in `python/infinicore/ops/<op>.py` and export it from `python/infinicore/__init__.py`.
11. Add an InfiniCore test in `test/infinicore/ops/<op>.py`.

## Descriptor Lifetime Rules

Sparse descriptors may store pointers to tensor descriptors and tensor data. Keep the owning `SpMat` alive for the whole operation.

Do not blindly use descriptor caches for sparse operators. If an InfiniOP descriptor stores an `infiniopSpMatDescriptor_t` from a temporary `SpMat`, caching that descriptor can leave dangling pointers after the `SpMat` is destroyed. SpMM avoids this by creating a descriptor per plan and storing the current `SpMat` inside the planned metadata.

## Build and Test

Common checks:

```shell
python scripts/format.py --check --path <changed-path>
xmake build infiniop-cpu
xmake build _infinicore
python test/infiniop/spmm.py --cpu
python test/infinicore/ops/spmm.py --cpu
```

For NVIDIA:

```shell
xmake f --nv-gpu=y --cuda=$CUDA_HOME -cv
xmake build infiniop-nvidia
python test/infiniop/spmm.py --nvidia
python test/infinicore/ops/spmm.py --nvidia
```

If Python imports an old extension or old shared library, rebuild and reinstall the extension or set `LD_LIBRARY_PATH` so tests load the freshly built `libinfinicore_cpp_api.so`, `libinfiniop.so`, and `_infinicore` module.
