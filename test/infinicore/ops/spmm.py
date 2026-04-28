"""
test/infinicore/ops/spmm.py — SpMM (Sparse Matrix × Dense Matrix) test.

This test mirrors the cuSPARSE-style workflow:

  1. Build a CSR sparse matrix A from dense data  →  ``infinicore.SpMat``
  2. Create a dense matrix B                      →  ``infinicore.Tensor``
  3. Compute C = alpha * A * B (+ beta * C)       →  ``infinicore.spmm`` / ``spmm_``
  4. Validate against a NumPy dense-matmul reference

The test runs on CPU and does not require any GPU hardware.

Usage::

    python test/infinicore/ops/spmm.py
"""

import sys
import os
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.sparse import SpMat


# ---------------------------------------------------------------------------
# Test cases: (m, k, n, density, alpha, beta)
# ---------------------------------------------------------------------------
_TEST_CASES = [
    (4,  4,  4,  0.50, 1.0, 0.0),
    (4,  4,  4,  0.50, 2.0, 1.0),  # non-trivial alpha and beta
    (8,  16, 4,  0.25, 1.0, 0.0),
    (1,  8,  8,  0.50, 1.0, 0.0),
    (32, 64, 16, 0.10, 1.0, 0.0),
    (16, 32, 8,  0.30, 0.5, 0.5),
]

_ATOL = 1e-4
_RTOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csr_matrix(dense: np.ndarray):
    """Return (row_offsets, col_indices, values) from a dense array."""
    m = dense.shape[0]
    rows_idx, cols_idx = np.where(dense != 0)
    values = dense[rows_idx, cols_idx].astype(np.float32)
    row_counts = np.bincount(rows_idx, minlength=m).astype(np.int32)
    row_offsets = np.zeros(m + 1, dtype=np.int32)
    np.cumsum(row_counts, out=row_offsets[1:])
    return row_offsets, cols_idx.astype(np.int32), values


def _tensor_to_numpy(t: infinicore.Tensor) -> np.ndarray:
    """Copy an infinicore CPU Tensor into a NumPy float32 array."""
    shape = list(t.shape)
    result = torch.zeros(shape, dtype=torch.float32, device="cpu")
    dest = infinicore.from_blob(
        result.data_ptr(), shape,
        dtype=infinicore.float32, device=infinicore.device("cpu", 0),
    )
    dest.copy_(t)
    return result.numpy()


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, atol=_ATOL, rtol=_RTOL))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests(device_type: str = "cpu", device_index: int = 0) -> None:
    device = infinicore.device(device_type, device_index)
    rng = np.random.default_rng(42)

    passed, failed = 0, 0

    for case_idx, (m, k, n, density, alpha, beta) in enumerate(_TEST_CASES):

        # ---- Build sparse matrix A (m × k) in CSR format ----------------
        dense_A = rng.random((m, k)).astype(np.float32)
        mask = (rng.random((m, k)) < density).astype(np.float32)
        dense_A *= mask
        if dense_A.max() == 0.0:   # guarantee at least one non-zero
            dense_A[0, 0] = 1.0

        row_offsets_np, col_indices_np, values_np = _make_csr_matrix(dense_A)
        sp_A = SpMat.from_csr(
            row_offsets_np, col_indices_np, values_np, m, k,
            device=device, dtype=infinicore.float32,
        )

        # ---- Build dense matrix B (k × n) --------------------------------
        B_np = rng.random((k, n)).astype(np.float32)
        B_torch = torch.from_numpy(B_np.copy())
        B_t = infinicore.from_blob(
            B_torch.data_ptr(), [k, n],
            dtype=infinicore.float32, device=device,
        )

        # ---- Reference: C_ref = alpha * dense_A @ B_np + beta * C_init --
        C_init_np = np.zeros((m, n), dtype=np.float32)
        if beta != 0.0:
            C_init_np = rng.random((m, n)).astype(np.float32)
        C_ref = alpha * (dense_A @ B_np) + beta * C_init_np

        # ---- InfiniCore spmm / spmm_ ------------------------------------
        if beta == 0.0:
            # Out-of-place: C = alpha * A * B
            C_t = infinicore.spmm(sp_A, B_t, alpha=alpha)
        else:
            # In-place:  C = alpha * A * B + beta * C
            C_init_torch = torch.from_numpy(C_init_np.copy())
            C_t = infinicore.from_blob(
                C_init_torch.data_ptr(), [m, n],
                dtype=infinicore.float32, device=device,
            )
            infinicore.spmm_(C_t, sp_A, B_t, alpha=alpha, beta=beta)

        C_out = _tensor_to_numpy(C_t)

        # ---- Compare -----------------------------------------------------
        ok = _allclose(C_out, C_ref)
        max_err = float(np.abs(C_out - C_ref).max())
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(
            f"[{status}] Case {case_idx}: m={m} k={k} n={n} "
            f"density={density} alpha={alpha} beta={beta} "
            f"| max_abs_err={max_err:.2e}"
        )

    print(f"\nResults: {passed}/{passed + failed} passed")
    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SpMM operator test")
    parser.add_argument("--device", default="cpu",
                        help="Device type, e.g. 'cpu' (default: cpu)")
    parser.add_argument("--device-index", type=int, default=0,
                        help="Device index (default: 0)")
    args = parser.parse_args()

    print(f"Running SpMM tests on device={args.device}:{args.device_index}\n")
    run_tests(device_type=args.device, device_index=args.device_index)


if __name__ == "__main__":
    main()
