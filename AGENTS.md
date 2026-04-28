# Repository Guidelines

## Project Structure & Module Organization

InfiniCore is a cross-platform compute stack with C APIs, C++ wrappers, Python bindings, and device-specific kernels. Public headers live in `include/`, especially `include/infinicore/` and `include/infiniop/`. Implementation code is under `src/`: `src/infinicore/` for the tensor framework, `src/infiniop/` for operator backends, `src/infinirt/` for runtime support, and `src/infiniccl/` for communication. Python package code is in `python/infinicore/`. Tests are split between `test/infinicore/` and `test/infiniop/`; generated InfiniOP test tooling is in `test/infiniop-test/`. Build configuration is rooted at `xmake.lua`, with per-platform options in `xmake/*.lua`.

## Build, Test, and Development Commands

- `source scripts/set_env_linux.sh`: set `INFINI_ROOT` and library paths for local development.
- `python scripts/install.py [XMAKE_CONFIG_FLAGS]`: configure, build, and install the low-level libraries.
- `xmake f -cv`: configure the default CPU build; add backend flags such as `--nv-gpu=y --cuda=$CUDA_HOME`.
- `xmake build && xmake install`: build and install configured targets.
- `xmake build _infinicore && xmake install _infinicore`: rebuild only the C++ InfiniCore library.
- `pip install -e .`: install the Python package in editable mode.

## Coding Style & Naming Conventions

Use C++17-compatible code. C/C++ and device sources are formatted with `clang-format-16` using `.clang-format`; Python uses `black`, and `pyproject.toml` enables basic `ruff` checks. Run `python scripts/format.py --check --path <path>` before submitting, or omit `--check` to format.

C++ types use `UpperCamelCase`; exported C pointer and enum types follow `infinixx[Xxx]_t`; constants use `INFINI_UPPER_SNAKE_CASE`; variables and parameters use `snake_case`; regular functions use `lowerCamelCase` unless matching Torch-style APIs, which use `snake_case`.

## Testing Guidelines

Run focused tests for the area changed. InfiniCore operator tests use:

```shell
python test/infinicore/ops/[operator].py --cpu
python test/infinicore/run.py --cpu
```

InfiniOP tests use:

```shell
python test/infiniop/[operator].py --cpu
python scripts/python_test.py --cpu
```

Replace `--cpu` with the relevant backend flag, such as `--nvidia`, `--cambricon`, `--ascend`, or `--kunlun`. For InfiniCCL, build `xmake build infiniccl-test` and run `infiniccl-test --nvidia` on supported hardware.

## Commit & Pull Request Guidelines

Follow the project’s issue-based workflow. Create branches named `issue/#` or `issue/#-description`. Commit messages and PR titles should start with `issue/#`, for example `issue/1126 fix softmax and conv2d`. Every PR should link the issue, describe the touched module or backend, list commands run, include final test evidence, and request at least two reviewers as described in `DEV.md`.

## Agent-Specific Instructions

Keep edits scoped to the requested module. Do not rewrite generated files, vendored dependencies, or unrelated backend implementations unless the task explicitly requires it.
