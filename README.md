# LinearAlgebra

[![CI](https://github.com/KhwarizmiAnalytix/LinearAlgebra/actions/workflows/ci.yml/badge.svg)](https://github.com/KhwarizmiAnalytix/LinearAlgebra/actions/workflows/ci.yml)
[![License: GPL v3 / Commercial](https://img.shields.io/badge/license-GPL--3.0--or--later%20%2F%20commercial-blue.svg)](LICENSE)

**Dense linear algebra**: Cholesky/LU/QR/SVD decompositions, linear solve,
least-squares (over/under-determined), matrix
inversion/multiplication/transpose/pseudo-inverse, symmetric and general
eigenvalue/eigenvector decomposition, and rank/condition-number/norm/trace
estimation, with a portable scalar fallback, an optional **Intel MKL**
(LAPACKE/BLAS) backend, an optional vendor-neutral **CBLAS/LAPACKE** backend
(OpenBLAS, Accelerate, netlib, ...), and an optional **cuBLAS/cuSOLVER** GPU backend
(`linalg::gpu::*`, device pointers only, no implicit host transfers)
covering every CPU op except general (non-symmetric) eigenvalue
decomposition, for which cuSOLVER's dense API has no `geev`-equivalent
routine to call into. See
[`docs/gpu_backend_design.md`](docs/gpu_backend_design.md) for the GPU
backend's design and current status.

Standalone CMake package — any C++ project can consume it via
`add_subdirectory`; [XSigma](https://github.com/KhwarizmiAnalytix/Hisab) is
one consumer, not a required host.

## Layout

- `CMakeLists.txt` — `LINALG_ENABLE_MKL`, `LINALG_ENABLE_BLAS`,
  `LINALG_ENABLE_CUBLAS`, `LINALG_LU_PIVOTING`, `LINALG_ENABLE_*`.
- `BUILD.bazel` — `//:LinearAlgebra` (CPU only — the GPU tree below is
  CMake-only, see `LINALG_ENABLE_CUBLAS` below).
- `include/common/` — export macro, feature macros, cuBLAS/cuSOLVER handle
  pool (`cuda_handle.h`).
- `include/memory/` — minimal scratch-buffer allocator.
- `include/util/` — exception helper, conservative CPU cache-size defaults.
- `include/matrix_operation/` — the public CPU modules (below).
- `include/matrix_operation_gpu/` — the `linalg::gpu::*` counterparts
  (below), built when `LINALG_ENABLE_CUBLAS=ON`.
- `Testing/Cxx/` — GoogleTest unit tests.
- `ThirdParty/` — vendored `googletest` submodule.
- `docs/` — design docs (GPU backend design and status).

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
| `matrix_operation/qr_decomposition.h` | Economy Householder QR (`A = Q*R`) |
| `matrix_operation/least_squares.h` | `least_squares_solve` — over/under-determined, via pseudo-inverse |
| `matrix_operation/pseudo_inverse.h` | Moore-Penrose pseudo-inverse, via SVD |
| `matrix_operation/matrix_rank.h` | `matrix_rank`, `matrix_condition_number`, via SVD |
| `matrix_operation/matrix_norm.h` | Frobenius / 1 / infinity / 2 (spectral) matrix norms |
| `matrix_operation/matrix_trace.h` | `matrix_trace` |
| `matrix_operation/eigenvalue_decomposition.h` | Symmetric (Jacobi) and general (real Schur QR) eigenvalue/eigenvector decomposition |

All entry points are plain row-major raw-pointer C++ (`T* data, lda`) — no
matrix class is part of the public API; callers own their own storage.

### GPU modules (`linalg::gpu::*`, `LINALG_ENABLE_CUBLAS=ON`)

| Header | Provides |
|--------|----------|
| `matrix_operation_gpu/cholesky_decomposition_gpu.h` | In-place Cholesky factorization on device memory |
| `matrix_operation_gpu/lu_decomposition_gpu.h` | In-place LU factorization, row-major "P·A = L·U" packed layout matching the CPU convention |
| `matrix_operation_gpu/linear_solver_gpu.h` | Cholesky/LU solve (no caller-visible pivot — cuSOLVER manages it internally) |
| `matrix_operation_gpu/matrix_inversion_gpu.h` | `matrix_invert`, `matrix_determinant` |
| `matrix_operation_gpu/matrix_multiplication_gpu.h` | cuBLAS GEMM with optional operand transpose |
| `matrix_operation_gpu/matrix_multiplication_batched_gpu.h` | Batched same-size square matrix multiply |
| `matrix_operation_gpu/matrix_transpose_gpu.h` | In-place transpose |
| `matrix_operation_gpu/svd_decomposition_gpu.h` | Economy SVD (tightly packed `lda`/`ldu`/`ldv` only — see the header) |
| `matrix_operation_gpu/qr_decomposition_gpu.h` | Economy Householder QR (tightly packed only) |
| `matrix_operation_gpu/eigenvalue_decomposition_gpu.h` | Symmetric eigenvalue/eigenvector decomposition only — cuSOLVER's dense API has no general (non-symmetric) `geev`-equivalent, so there is no GPU counterpart for that case; use the CPU `eigenvalue_decomposition` |
| `matrix_operation_gpu/pseudo_inverse_gpu.h` | Moore-Penrose pseudo-inverse, via GPU SVD |
| `matrix_operation_gpu/matrix_rank_gpu.h` | `matrix_rank`, `matrix_condition_number`, via GPU SVD (downloads only the small singular-value vector, never the full matrix) |
| `matrix_operation_gpu/matrix_norm_gpu.h` | Frobenius / 1 / infinity / 2 matrix norms (`cublasXnrm2`/`cublasIXamax` plus a small reduction kernel for 1/infinity) |
| `matrix_operation_gpu/matrix_trace_gpu.h` | `matrix_trace` |
| `matrix_operation_gpu/least_squares_gpu.h` | `least_squares_solve` — over/under-determined, via GPU pseudo-inverse |

Every GPU entry point takes device pointers only (never touches host
memory internally) and an optional trailing `cudaStream_t` (default: the
legacy default stream); `matrix_determinant`/`pseudo_inverse`/`matrix_rank`/
`matrix_condition_number`/`matrix_norm`/`matrix_trace` are the exceptions
that must synchronize once and copy back a single scalar (or, for
`pseudo_inverse`, the small singular-value vector) to compute their result
— every other buffer involved stays device-resident. See
[`docs/gpu_backend_design.md`](docs/gpu_backend_design.md) for the backend's
design, known restrictions, and what's still pending (multi-RHS solve,
batched factorizations).

## CMake options

| CMake variable | Default | Summary |
|-----------------|---------|---------|
| `LINALG_ENABLE_MKL` | OFF | MKL-backed LAPACKE/BLAS code paths in every CPU module that has one directly (cholesky/lu/svd/qr/inversion/multiplication/transpose/linear_solver/eigenvalue_decomposition); least_squares/pseudo_inverse/matrix_rank/matrix_norm/matrix_trace compose on top of svd_decomposition and pick up the same backend transitively |
| `LINALG_ENABLE_BLAS` | OFF | Vendor-neutral CBLAS/LAPACKE backend (OpenBLAS, Accelerate, netlib, ...); MKL takes precedence when both are ON |
| `LINALG_ENABLE_CUBLAS` | OFF | cuBLAS/cuSOLVER GPU backend (`linalg::gpu::*`, `include/matrix_operation_gpu/`); requires the CUDA toolkit, enables the CUDA language only when ON |
| `LINALG_LU_PIVOTING` | OFF | Partial pivoting in the scalar (non-MKL) LU fallback paths |
| `LINALG_DISPLAY_WIN32_WARNINGS` | OFF | Re-enable MSVC narrowing-conversion warnings around the scalar fallback |
| `LINALG_ENABLE_TESTING` | ON | Build the test suite |
| `LINALG_ENABLE_GTEST` | ON | GoogleTest integration for tests |
| `LINALG_CXX_STANDARD` | 17 | `17`, `20`, or `23` |
| Other `LINALG_ENABLE_*` | see `CMakeLists.txt` | LTO, coverage, sanitizers, cache, clang-tidy, IWYU, valgrind, … |

When `LINALG_ENABLE_MKL` is `ON`, `find_package(MKL)` must resolve (via
`MKLROOT` / `CMAKE_PREFIX_PATH`) or configuration fails with `FATAL_ERROR`.
When `LINALG_ENABLE_BLAS` is `ON`, `find_package(BLAS)`/`find_package(LAPACK)`
must resolve, or configuration fails with `FATAL_ERROR`; `cblas.h` is
required, but `lapacke.h` is optional — without it, the backend covers only
`matrix_multiplication`/`matrix_transpose` and every other op falls back to
scalar (a `STATUS` message reports which case applies). When
`LINALG_ENABLE_CUBLAS` is `ON`, `find_package(CUDAToolkit)` must resolve.

## Public API (abridged)

```cpp
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver.h"

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
discoverable by `find_package(MKL)`), the vendor-neutral BLAS backend with
`-DLINALG_ENABLE_BLAS=ON`, or the GPU backend with `-DLINALG_ENABLE_CUBLAS=ON`
(requires the CUDA toolkit; set `CMAKE_CUDA_ARCHITECTURES` for your GPU, or
leave it to `native` detection).

## Bazel

```bash
bazel test //...
```

Bazel version is pinned via [`.bazelversion`](.bazelversion): `WORKSPACE.bazel`
wires `ThirdParty/googletest` in via `local_repository`.

## CI & Coverage

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
PR to `main`: CMake scalar-fallback build+test on Linux and macOS, a CMake
build+test with `-DLINALG_ENABLE_MKL=ON` (Linux, Intel oneMKL via apt), and
`bazel test //...` on Linux and macOS. `LINALG_ENABLE_BLAS` and
`LINALG_ENABLE_CUBLAS` are **not** covered by CI (no BLAS/LAPACKE job; no
GPU runner) — both are exercised only by local, manual builds; see
[`docs/gpu_backend_design.md`](docs/gpu_backend_design.md) §7 for how the
GPU backend is verified today. See [CONTRIBUTING.md](CONTRIBUTING.md) for
running the same checks locally, plus sanitizers, coverage, and lintrunner
(not yet wired into CI).
