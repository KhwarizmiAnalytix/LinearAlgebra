# LinearAlgebra

[![License: GPL v3 / Commercial](https://img.shields.io/badge/license-GPL--3.0--or--later%20%2F%20commercial-blue.svg)](LICENSE)

**Dense linear algebra**: Cholesky/LU/SVD decompositions, linear solve, and
matrix inversion/multiplication/transpose, with a portable scalar fallback
and an optional **Intel MKL** (LAPACKE/BLAS) backend.

Standalone CMake package — any C++ project can consume it via
`add_subdirectory`; [XSigma](https://github.com/KhwarizmiAnalytix/Hisab) is
one consumer, not a required host.

## Layout

- `CMakeLists.txt` — `LINALG_ENABLE_MKL`, `LINALG_LU_PIVOTING`, `LINALG_ENABLE_*`.
- `BUILD.bazel` — `//:LinearAlgebra`.
- `linear_algebra/common/` — export macro, feature macros, generated `configure.h`.
- `linear_algebra/memory/` — minimal scratch-buffer allocator.
- `linear_algebra/util/` — exception helper, conservative CPU cache-size defaults.
- `linear_algebra/matrix_operation/` — the seven public modules (below).
- `Testing/Cxx/` — GoogleTest unit tests.
- `ThirdParty/` — vendored `googletest` submodule.

Public C++ namespace: `linalg`. Macros: `LINALG_API`, `LINALG_THROW`, `LINALG_UNUSED`.

## Modules

| Header | Provides |
|--------|----------|
| `matrix_operation/cholesky_decomposition.h` | In-place Cholesky factorization (upper/lower), scalar or MKL LAPACKE |
| `matrix_operation/lu_decomposition.h` | In-place LU factorization with pivot vector |
| `matrix_operation/linear_solver.h` | `linear_solver_type`-selectable solve (LU / Cholesky, upfront or fused) |
| `matrix_operation/matrix_inversion.h` | `matrix_invert`, `matrix_determinant` |
| `matrix_operation/matrix_multiplication.h` | Row-major GEMM with optional operand transpose |
| `matrix_operation/matrix_transpose.h` | In-place / out-of-place transpose |
| `matrix_operation/svd_decomposition.h` | Singular value decomposition |

All entry points are plain row-major raw-pointer C++ (`T* data, lda`) — no
matrix class is part of the public API; callers own their own storage.

## CMake options

| CMake variable | Default | Summary |
|-----------------|---------|---------|
| `LINALG_ENABLE_MKL` | OFF | MKL-backed LAPACKE/BLAS code paths in all seven modules |
| `LINALG_LU_PIVOTING` | OFF | Partial pivoting in the scalar (non-MKL) LU fallback paths |
| `LINALG_DISPLAY_WIN32_WARNINGS` | OFF | Re-enable MSVC narrowing-conversion warnings around the scalar fallback |
| `LINALG_ENABLE_TESTING` | ON | Build the test suite |
| `LINALG_ENABLE_GTEST` | ON | GoogleTest integration for tests |
| `LINALG_CXX_STANDARD` | 17 | `17`, `20`, or `23` |
| Other `LINALG_ENABLE_*` | see `CMakeLists.txt` | LTO, coverage, sanitizers, cache, clang-tidy, IWYU, valgrind, … |

When `LINALG_ENABLE_MKL` is `ON`, `find_package(MKL)` must resolve (via
`MKLROOT` / `CMAKE_PREFIX_PATH`) or configuration fails with `FATAL_ERROR`.

## Public API (abridged)

```cpp
#include "linear_algebra/matrix_operation/cholesky_decomposition.h"
#include "linear_algebra/matrix_operation/linear_solver.h"

// In-place lower-triangular Cholesky factor of an n x n row-major matrix.
linalg::cholesky_decomposition(A.data(), n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);

// Solve A x = b in place, reusing the Cholesky factor above.
linalg::linear_solver(
    A.data(), pivot.data(), n, x.data(),
    linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);
```

## Building

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Enable the MKL backend with `-DLINALG_ENABLE_MKL=ON` (requires Intel oneMKL
discoverable by `find_package(MKL)`).

## Bazel

```bash
bazel test //...
```

Bazel version is pinned via [`.bazelversion`](.bazelversion): `WORKSPACE.bazel`
wires `ThirdParty/googletest` in via `local_repository`.

## CI & Coverage

See [CONTRIBUTING.md](CONTRIBUTING.md) for running the same checks locally
(build, sanitizers, coverage, lintrunner).
