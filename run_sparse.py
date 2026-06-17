#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

if "INFINICORE_SPARSE_MTX_DIR" in os.environ:
    del os.environ["INFINICORE_SPARSE_MTX_DIR"]

SPARSE_OPS = [
    "spmv",
    "spmm",
    "sddmm",
    "axpby",
    "sparse_gather",
    "sparse_scatter",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sparse InfiniCore operator tests with report saving."
    )
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--metax", action="store_true", help="Run on Metax")
    backend.add_argument("--cambricon", action="store_true", help="Run on Cambricon")
    return parser.parse_args()


def main():
    args = parse_args()
    backend_arg = "--metax" if args.metax else "--cambricon"
    command = [
        sys.executable,
        "test/infinicore/run.py",
        "--ops",
        *SPARSE_OPS,
        "--save",
        backend_arg,
    ]
    return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
