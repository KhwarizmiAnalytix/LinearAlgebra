# LinearAlgebra

[![CI](https://github.com/KhwarizmiAnalytix/LinearAlgebra/actions/workflows/ci.yml/badge.svg)](https://github.com/KhwarizmiAnalytix/LinearAlgebra/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KhwarizmiAnalytix/LinearAlgebra/branch/main/graph/badge.svg)](https://codecov.io/gh/KhwarizmiAnalytix/LinearAlgebra)
[![License: GPL v3 / Commercial](https://img.shields.io/badge/license-GPL--3.0--or--later%20%2F%20commercial-blue.svg)](LICENSE)

**Dense linear algebra**: Cholesky/LU/QR/SVD decompositions, linear solve,
least-squares (over/under-determined), matrix
inversion/multiplication/transpose/pseudo-inverse, symmetric and general
eigenvalue/eigenvector decomposition, and rank/condition-number/norm/trace
estimation, with a portable scalar fallback, an optional **Intel MKL**
(LAPACKE/BLAS) backend, an optional vendor-neutral **CBLAS/LAPACKE** backend
(OpenBLAS, Accelerate, netlib, ...), and an optional **cuBLAS/cuSOLVER** GPU backend
(same struct-template API, `linalg_backend::cuda_blas` specializations,
device pointers only, no implicit host transfers) covering every CPU op
except general (non-symmetric) eigenvalue decomposition, for which
cuSOLVER's dense API has no `geev`-equivalent routine to call into. Error reporting always goes through
[KhwarizmiAnalytix/Logging](https://github.com/KhwarizmiAnalytix/Logging)
(vendored at `ThirdParty/Logging`, a required dependency, not an optional
backend). See [`docs/gpu_backend_design.md`](docs/gpu_backend_design.md)
for the GPU backend's design and current status.

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
- `include/util/` — conservative CPU cache-size defaults (`cpu_info.h`); the
  exception helper that used to live here is gone — error reporting goes
  straight through Logging's `LOGGING_THROW`/`logging::exception` now.
- `include/matrix_operation/` — the public modules (CPU and GPU); the GPU
  `linalg_backend::cuda_blas` specializations are in
  `src/matrix_operation/cublas/`, built when `LINALG_ENABLE_CUBLAS=ON`.
- `src/` — implementation (`.cpp`/`.cu`) files, mirroring `include/`'s
  layout (e.g. `include/matrix_operation/lu_decomposition.h` pairs with
  `src/matrix_operation/native/lu_decomposition_native.cpp` on the scalar
  path and `src/matrix_operation/cublas/lu_decomposition_cublas.cpp` on the
  GPU path).
- `Testing/Cxx/` — GoogleTest unit tests.
- `ThirdParty/` — vendored `googletest` submodule, plus `Logging`
  ([KhwarizmiAnalytix/Logging](https://github.com/KhwarizmiAnalytix/Logging)),
  a required dependency (always built, not gated by an option).
- `docs/` — design docs (GPU backend design and status).

Public C++ namespace: `linalg`. Macros: `LINALG_API`, `LINALG_UNUSED`; error
reporting uses Logging's own `LOGGING_THROW`/`LOGGING_CHECK` directly (fmt-style
`{}` placeholders) and throws/catches `logging::exception`.

## Modules

| Header | Provides |
|--------|----------|
| `matrix_operation/cholesky_decomposition.h` | In-place Cholesky factorization (upper/lower), scalar or MKL LAPACKE |
| `matrix_operation/lu_decomposition.h` | In-place LU factorization with pivot vector |
| `matrix_operation/linear_solver.h` | `linear_solver_type`-selectable solve (LU / Cholesky, upfront or fused) |
| `matrix_operation/inversion.h` | `matrix_invert` |
| `matrix_operation/determinant.h` | `determinant` |
| `matrix_operation/multiplication.h` | Row-major GEMM with optional operand transpose |
| `matrix_operation/transpose.h` | In-place / out-of-place transpose |
| `matrix_operation/svd_decomposition.h` | Singular value decomposition |
| `matrix_operation/qr_decomposition.h` | Economy Householder QR (`A = Q*R`) |
| `matrix_operation/least_squares.h` | `least_squares_solve` — over/under-determined, via pseudo-inverse |
| `matrix_operation/pseudo_inverse.h` | Moore-Penrose pseudo-inverse, via SVD |
| `matrix_operation/rank.h` | `rank`, via SVD |
| `matrix_operation/condition_number.h` | `condition_number`, via SVD |
| `matrix_operation/norm.h` | Frobenius / 1 / infinity / 2 (spectral) matrix norms |
| `matrix_operation/trace.h` | `trace` |
| `matrix_operation/eigen.h` | Symmetric (Jacobi) and general (real Schur QR) eigenvalue/eigenvector decomposition |
| `matrix_operation/exponential.h` | `exponential` via scaling and squaring with a Padé approximant, composed on `multiplication`/`matrix_invert`/`norm` (inherits their backend transitively) |
| `matrix_operation/tridiagonal_operations.h` | Tridiagonal solve/multiply/decompose — scalar only, `std::vector`-based API outside the backend model |
| `matrix_operation/pentadiagonal_operations.h` | Pentadiagonal solve/multiply — scalar only, `std::vector`-based API outside the backend model |

All entry points are plain row-major raw-pointer C++ (`T* data, lda`) — no
matrix class is part of the public API; callers own their own storage.

### GPU backend (`linalg_backend::cuda_blas`, `LINALG_ENABLE_CUBLAS=ON`)

The GPU backend reuses the same struct-template API as every other backend.
Include the same `include/matrix_operation/*.h` headers as the CPU path and
use `linalg_backend::cuda_blas` as the template argument:

```cpp
#include "include/matrix_operation/svd_decomposition.h"

// Query workspace, allocate on device, then call:
linalg::svd_decomposition<double, linalg_backend::cuda_blas>::apply(
    rows, cols, dA, lda, dS, dU, ldu, dVT, ldv, d_workspace, ws_size);
```

All entry points take device pointers only (never touch host memory
internally); most accept an optional trailing `void* stream` (cast to
`cudaStream_t` internally; default: `nullptr` for the legacy default stream).
Operations that return a scalar (`trace`, `determinant`, `norm`, `rank`,
`condition_number`) synchronize internally and return the host value directly.
`cholesky_decomposition<T, cuda_blas>::apply` and
`trace<T, cuda_blas>::apply` allocate their own workspace internally (the
struct `apply` has no workspace parameter for those two operations).

For batched square-matrix multiplication, include
`include/matrix_operation/multiplication_batched.h` and call
`linalg::batched_matrix_multiplication(dim, count, a, b, c, stream)`.

See [`docs/gpu_backend_design.md`](docs/gpu_backend_design.md) for the
backend's design and known restrictions.

## CMake options

| CMake variable | Default | Summary |
|-----------------|---------|---------|
| `LINALG_ENABLE_MKL` | OFF | MKL-backed LAPACKE/BLAS code paths in every CPU module that has one directly (cholesky/lu/svd/qr/inversion/multiplication/transpose/linear_solver/eigen); least_squares/pseudo_inverse/rank/norm/trace compose on top of svd_decomposition and pick up the same backend transitively |
| `LINALG_ENABLE_BLAS` | OFF | Vendor-neutral CBLAS/LAPACKE backend (OpenBLAS, Accelerate, netlib, ...); MKL takes precedence when both are ON |
| `LINALG_ENABLE_CUBLAS` | OFF | cuBLAS/cuSOLVER GPU backend (`linalg_backend::cuda_blas` specializations, `src/matrix_operation/cublas/`); requires the CUDA toolkit, enables the CUDA language only when ON |
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
`multiplication`/`transpose` and every other op falls back to
scalar (a `STATUS` message reports which case applies). When
`LINALG_ENABLE_CUBLAS` is `ON`, `find_package(CUDAToolkit)` must resolve.

## Backend landscape

General reference for where each vendor backend sits (CPU/GPU, which
operation families it covers); not every row is wired into LinearAlgebra —
see the Implemented column.

| Backend | CPU/GPU | Dense BLAS | LAPACK | Sparse | FFT | Implemented here |
|---|---|---:|---:|---:|---:|---|
| **Intel oneMKL** | CPU + Intel GPU | ✅ | ✅ | ✅ | ✅ | ✅ `LINALG_ENABLE_MKL` |
| **OpenBLAS** | CPU | ✅ | ✅* | ❌ | ❌ | ✅ via `LINALG_ENABLE_BLAS` |
| **Apple Accelerate** | CPU / Apple Silicon | ✅ | ✅ | ✅ | ✅ | ✅ via `LINALG_ENABLE_BLAS` |
| **Netlib BLAS/LAPACK** | CPU | ✅ | ✅ | ❌ | ❌ | ✅ via `LINALG_ENABLE_BLAS` |
| **cuBLAS / cuSOLVER** | NVIDIA GPU | ✅ | ✅ | ❌ | ❌ | ✅ `LINALG_ENABLE_CUBLAS` (`linalg_backend::cuda_blas`) |
| **BLIS** | CPU | ✅ | ❌ | ❌ | ❌ | ❌ not wired up |
| **cuBLASLt** | NVIDIA GPU | ✅ | — | — | — | ❌ not wired up |
| **cuSPARSE** | NVIDIA GPU | — | — | ✅ | — | ❌ out of scope (dense only) |
| **rocBLAS** | AMD GPU | ✅ | — | — | — | ❌ not wired up |
| **hipSOLVER** | AMD GPU | — | ✅ | — | — | ❌ not wired up |
| **rocSPARSE** | AMD GPU | — | — | ✅ | — | ❌ out of scope (dense only) |

\* OpenBLAS ships LAPACK only when built with its bundled reference LAPACK
(the common case for prebuilt packages); a BLAS-only OpenBLAS build falls
back to the scalar path here for LAPACK-level ops (see `LINALG_ENABLE_BLAS`
above). FFT and sparse routines are out of scope for LinearAlgebra, which is
dense-only; the FFT/Sparse columns above describe each vendor library in
general, not something this repo calls.

## Public API (abridged)

```cpp
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver.h"

// In-place lower-triangular Cholesky factor of an n x n row-major matrix.
// Choose native, blas_lapacke, or mkl explicitly for each operation.
linalg::cholesky_decomposition<double, linalg::linalg_backend::native>::apply(
    A.data(), n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);

// Solve A x = b in place, reusing the Cholesky factor above.
linalg::linear_solver<double, linalg::linalg_backend::native>::apply(
    A.data(), pivot.data(), n, x.data(),
    linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER, nullptr, 0);
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
PR to `main`: native CMake builds with GCC and Clang, OpenBLAS/LAPACKE and
MKL builds, LU pivoting, sanitizers, and `bazel test //Testing/...`.
The SIMD matrix covers vectorization disabled and each explicit ISA; an ISA
is compiled but not executed when the runner lacks the required instructions.
Shared and static install-consumer jobs build a separate application against
the installed packages. Coverage runs on Linux/macOS/Windows for the native
backend and on Linux for OpenBLAS/LAPACKE and MKL, with separate report artifacts
and Codecov flags per backend.

CUDA is always compiled in a GPU-less container. GPU runtime tests run on pushes
to `main` when the repository variable `LINALG_GPU_CI_ENABLED` is `true` and a
self-hosted runner with labels `linux`, `x64`, and `gpu` is registered. That runner
needs Docker, a compatible NVIDIA driver, and the NVIDIA Container Toolkit;
the job installs its build tools inside the CUDA container. Until that runner
is available, GPU numerical correctness requires manual hardware testing.
See [CONTRIBUTING.md](CONTRIBUTING.md)
for running the same checks locally, plus sanitizers and lintrunner (not yet
wired into CI).
