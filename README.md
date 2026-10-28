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
(OpenBLAS, Accelerate, netlib, ...), and an optional **cuBLAS/cuSOLVER**
(NVIDIA) or **hipBLAS/hipSOLVER** (AMD) GPU backend — mutually exclusive,
same struct-template API, `linalg_backend::CUDA_BLAS`/`ROCM_BLAS`
specializations, device pointers only, no implicit host transfers —
covering every CPU op except general (non-symmetric) eigenvalue
decomposition, for which neither cuSOLVER's nor hipSOLVER's dense API has
a `geev`-equivalent routine to call into. Error reporting always goes through
[KhwarizmiAnalytix/Logging](https://github.com/KhwarizmiAnalytix/Logging)
(vendored at `ThirdParty/Logging`, a required dependency, not an optional
backend). See [`docs/gpu_backend_design.md`](docs/gpu_backend_design.md)
for the GPU backend's design and current status.

Standalone CMake package — any C++ project can consume it via
`add_subdirectory`; [XSigma](https://github.com/KhwarizmiAnalytix/Hisab) is
one consumer, not a required host.

## Layout

- `CMakeLists.txt` — `LINALG_ENABLE_MKL`, `LINALG_ENABLE_BLAS`,
  `LINALG_ENABLE_CUBLAS`, `LINALG_ENABLE_ROCBLAS` (mutually exclusive with
  `LINALG_ENABLE_CUBLAS`), `LINALG_LU_PIVOTING`, `LINALG_ENABLE_*`.
- `BUILD.bazel` — `//:LinearAlgebra` (CPU only — the GPU trees below are
  CMake-only, see `LINALG_ENABLE_CUBLAS`/`LINALG_ENABLE_ROCBLAS` below).
- `include/common/` — export macro, feature macros, cuBLAS/cuSOLVER handle
  pool (`cuda_handle.h`) and hipBLAS/hipSOLVER handle pool (`hip_handle.h`).
- `include/memory/` — minimal scratch-buffer allocator.
- `include/util/` — conservative CPU cache-size defaults (`cpu_info.h`); the
  exception helper that used to live here is gone — error reporting goes
  straight through Logging's `LOGGING_THROW`/`logging::exception` now.
- `include/matrix_operation/` — the public modules (CPU and GPU); the GPU
  `linalg_backend::CUDA_BLAS` specializations are in
  `src/matrix_operation/gpu/cublas/` (built when `LINALG_ENABLE_CUBLAS=ON`),
  and `linalg_backend::ROCM_BLAS` specializations are in
  `src/matrix_operation/gpu/hipblas/` (built when `LINALG_ENABLE_ROCBLAS=ON`).
- `src/` — implementation (`.cpp`/`.cu`/`.hip`) files, mirroring `include/`'s
  layout (e.g. `include/matrix_operation/lu_decomposition.h` pairs with
  `src/matrix_operation/cpu/native/lu_decomposition_native.cpp` on the scalar
  path, `src/matrix_operation/gpu/cublas/lu_decomposition_cublas.cpp` on the
  CUDA path, and `src/matrix_operation/gpu/hipblas/lu_decomposition_hip.cpp`
  on the ROCm path).
- `Testing/Cxx/` — GoogleTest unit tests.
- `ThirdParty/` — vendored `Scalars`, `eigen`, and `googletest` submodules, plus `Logging`
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
| `matrix_operation/triangular_solver.h` | Multiple-right-hand-side triangular solve; upper/lower, transpose, and unit-diagonal modes; native SIMD/GEMM, BLAS/MKL/cuBLAS/hipBLAS TRSM, or Metal MPS |
| `matrix_operation/inversion.h` | `matrix_invert` |
| `matrix_operation/determinant.h` | `determinant` |
| `matrix_operation/multiplication.h` | Row-major GEMM with optional operand transpose |
| `matrix_operation/transpose.h` | In-place / out-of-place transpose |
| `matrix_operation/svd_decomposition.h` | Singular value decomposition |
| `matrix_operation/qr_decomposition.h` | Economy Householder QR (`A = Q*R`) |
| `matrix_operation/schur_decomposition.h` | Complex Schur decomposition (`A = Q*S*Q^*`, S complex upper triangular); all six backends (Metal: float only); used by `logarithm`/`square_root` |
| `matrix_operation/least_squares.h` | `least_squares_solve` — over/under-determined, via pseudo-inverse |
| `matrix_operation/pseudo_inverse.h` | Moore-Penrose pseudo-inverse, via SVD |
| `matrix_operation/rank.h` | `rank`, via SVD |
| `matrix_operation/condition_number.h` | `condition_number`, via SVD |
| `matrix_operation/norm.h` | Frobenius / 1 / infinity / 2 (spectral) matrix norms |
| `matrix_operation/trace.h` | `trace` |
| `matrix_operation/eigen.h` | Symmetric (Householder/QL) and general (Schur QR) eigenvalue/eigenvector decomposition |
| `matrix_operation/exponential.h` | `exponential` via scaling and squaring with a Padé approximant, composed on `multiplication`/`matrix_invert`/`norm` (inherits their backend transitively) |
| `matrix_operation/logarithm.h` | `logarithm` — principal real matrix logarithm using Schur reduction and inverse scaling; all six backends (Metal: float only) |
| `matrix_operation/square_root.h` | `square_root` — principal real matrix square root using Schur reduction; all six backends (Metal: float only) |
| `matrix_operation/tridiagonal_operations.h` | Tridiagonal solve/multiply/decompose — scalar only, `std::vector`-based API outside the backend model |
| `matrix_operation/pentadiagonal_operations.h` | Pentadiagonal solve/multiply — scalar only, `std::vector`-based API outside the backend model |

All entry points are plain row-major raw-pointer C++ (`T* data, lda`) — no
matrix class is part of the public API; callers own their own storage.

`logarithm<T, Backend>` and `square_root<T, Backend>` follow the same
`apply(A, n, lda, result, ldc, workspace, bytes, stream)` and
`workspace_size(n)` interface as `exponential`. These are matrix functions;
for example, `square_root(A) * square_root(A) = A`. See
[matrix function algorithms and backend behavior](docs/matrix_functions.md)
for domain restrictions, source attribution, and performance limits.
See [triangular solver](docs/triangular_solver.md) for multiple-RHS solve modes and backend behavior.

### GPU backend (`linalg_backend::CUDA_BLAS`/`ROCM_BLAS`, `LINALG_ENABLE_CUBLAS=ON`/`LINALG_ENABLE_ROCBLAS=ON`)

The GPU backend reuses the same struct-template API as every other backend.
Include the same `include/matrix_operation/*.h` headers as the CPU path and
use `linalg_backend::CUDA_BLAS` (NVIDIA, cuBLAS/cuSOLVER) or
`linalg_backend::ROCM_BLAS` (AMD, hipBLAS/hipSOLVER) as the template
argument — the two options are mutually exclusive in a given build:

```cpp
#include "include/matrix_operation/svd_decomposition.h"

// Query workspace, allocate on device, then call:
linalg::svd_decomposition<double, linalg::linalg_backend::CUDA_BLAS>::apply(
    rows, cols, dA, lda, dS, dU, ldu, dVT, ldv, d_workspace, ws_size);
// Or, in a LINALG_ENABLE_ROCBLAS build:
linalg::svd_decomposition<double, linalg::linalg_backend::ROCM_BLAS>::apply(
    rows, cols, dA, lda, dS, dU, ldu, dVT, ldv, d_workspace, ws_size);
```

All GPU matrix, output and workspace pointers refer to device memory. Calls
accept an optional trailing `void* stream` (default `nullptr`) and perform no
host/device transfers or explicit stream synchronization. Scalar operations
(`trace`, `determinant`, `norm`, `rank`, `condition_number`) write to a required
device result pointer before the stream argument. LU, Cholesky and symmetric
eigen write numerical status to a required device `int* info`; LU also writes
a caller-owned device `int* pivot`. Cholesky takes caller-owned workspace,
queried with `workspace_size(n, triangle)`. CPU signatures are unchanged.

```cpp
linalg::norm<double, linalg::linalg_backend::CUDA_BLAS>::apply(
    dA, rows, cols, cols, linalg::norm_type::FROBENIUS,
    nullptr, 0, d_result, stream);
```

The caller owns buffer lifetimes and any deliberate upload/download at the
application boundary. GPU scalar APIs no longer return host values.

For batched square-matrix multiplication, include
`include/matrix_operation/multiplication_batched.h` and call
`linalg::batched_matrix_multiplication(dim, count, a, b, c, stream)`.

See [`docs/gpu_backend_design.md`](docs/gpu_backend_design.md) for the
backend's design and known restrictions.

## CMake options

CMake 3.24 or newer and C++20 are required by the Scalars dependency.
C++20 is the default; C++23 is also supported. The exported target publishes
its required language standard. CI exercises the C++20 baseline on Linux,
macOS, and Windows with bundled dependencies and the scalar algorithm path.

[Scalars](https://github.com/KhwarizmiAnalytix/Scalars), pinned in
`ThirdParty/Scalars`, supplies complex arithmetic and the `simd<T>` CPU
primitives. The native kernels and complex vector operations use that API
directly. `LINALG_CPU_BACKEND` selects the Scalars backend (`no`, `sse`,
`avx`, `avx2`, `avx512`, `neon`, or `sve`). `no` selects scalar loops.
Runtime-compiled Metal kernels retain an MSL compatibility layer.

| CMake variable | Default | Summary |
|-----------------|---------|---------|
| `LINALG_ENABLE_MKL` | OFF | MKL-backed LAPACKE/BLAS code paths in every CPU module that has one directly (cholesky/lu/svd/qr/inversion/multiplication/transpose/linear_solver/eigen); least_squares/pseudo_inverse/rank/norm/trace compose on top of svd_decomposition and pick up the same backend transitively |
| `LINALG_ENABLE_BLAS` | OFF | Vendor-neutral CBLAS/LAPACKE backend (OpenBLAS, Accelerate, netlib, ...); uses an independent shared provider when MKL is also ON |
| `LINALG_ENABLE_BLIS` | OFF | Select BLIS as the CBLAS provider for `BLAS_LAPACKE`; activates BLAS support and requires a paired LAPACKE library for decompositions |
| `LINALG_BLAS_RUNTIME_LIBRARY` | auto-detected OpenBLAS | Shared CBLAS provider (`.dll`, `.so`, `.dylib`) for combined MKL/BLAS builds |
| `LINALG_LAPACKE_RUNTIME_LIBRARY` | empty | Optional separate shared LAPACKE provider; empty means LAPACKE is bundled in the CBLAS provider |
| `LINALG_ENABLE_CUBLAS` | OFF | cuBLAS/cuSOLVER GPU backend (`linalg_backend::CUDA_BLAS` specializations, `src/matrix_operation/gpu/cublas/`); requires the CUDA toolkit, enables the CUDA language only when ON; mutually exclusive with ROCm and Metal |
| `LINALG_ENABLE_ROCBLAS` | OFF | hipBLAS/hipSOLVER GPU backend (`linalg_backend::ROCM_BLAS` specializations, `src/matrix_operation/gpu/hipblas/`); requires a ROCm/HIP toolchain, enables the HIP language only when ON; mutually exclusive with CUDA and Metal |
| `LINALG_ENABLE_METAL` | OFF | Metal Performance Shaders GPU backend (`linalg_backend::METAL_BLAS`); Apple platforms, float only; enables Objective-C++ and links Metal, MetalPerformanceShaders, and Foundation; mutually exclusive with CUDA and ROCm |
| `LINALG_BUILD_EXAMPLES` | OFF | Build the four-backend CPU/GPU benchmark example; requires BLAS/LAPACKE, MKL and cuBLAS or hipBLAS |
| `LINALG_ENABLE_PROFILING` | OFF | Enable Profiler instrumentation and build profiling programs; requires `Profiler::Profiler`. When OFF, Profiler is neither discovered nor linked |
| `LINALG_CPU_BACKEND` | `sse` on x86_64, `neon` on arm64, `no` otherwise | CPU SIMD backend: `no`, `sse`, `avx`, `avx2`, `avx512`, `neon`, `sve`. `no` is scalar loops |
| `LINALG_LU_PIVOTING` | OFF | Partial pivoting in the scalar (non-MKL) LU fallback paths |
| `LINALG_DISPLAY_WIN32_WARNINGS` | OFF | Re-enable MSVC narrowing-conversion warnings around the scalar fallback |
| `LINALG_ENABLE_TESTING` | Standalone: ON; embedded: host setting or OFF | Build the test suite; `BUILD_TESTING=OFF` disables it |
| `LINALG_ENABLE_BENCHMARK` | OFF | Build Google Benchmark micro-benchmarks |
| `LINALG_ENABLE_GTEST` | ON | Build GoogleTest cases; OFF retains the GPU source check without GoogleTest |
| `LINALG_CXX_STANDARD` | 20 | `20` or `23` |
| Other `LINALG_ENABLE_*` | see `CMakeLists.txt` | LTO, coverage, sanitizers, cache, clang-tidy, IWYU, valgrind, … |

When `LINALG_ENABLE_MKL` is `ON`, `find_package(MKL CONFIG)` must provide
`MKL::MKL` with its ILP64 interface (via `MKLROOT` / `CMAKE_PREFIX_PATH`).
Variable-only classic FindMKL modules are unsupported.
When only `LINALG_ENABLE_BLAS` is `ON`, `find_package(BLAS)`/`find_package(LAPACK)`
must resolve, or configuration fails with `FATAL_ERROR`; `cblas.h` is
required. Compile-and-link probes validate the LP64 CBLAS interface and enable
LAPACKE only when matching declarations and symbols are available. Accelerate
always uses its own headers and leaves LAPACKE disabled. Without LAPACKE, the backend covers only
`multiplication`/`transpose` and every other op falls back to
scalar (a `STATUS` message reports which case applies). When
`LINALG_ENABLE_CUBLAS` is `ON`, `find_package(CUDAToolkit)` must resolve.
When `LINALG_ENABLE_ROCBLAS` is `ON`, `find_package(hip)`,
`find_package(hipblas)` and `find_package(hipsolver)` must all resolve.
Configuration fails with `FATAL_ERROR` if more than one of `LINALG_ENABLE_CUBLAS`,
`LINALG_ENABLE_ROCBLAS`, and `LINALG_ENABLE_METAL` is `ON` at once.

SIMD defaults to SSE on x86_64, NEON on arm64, and scalar (`no`) elsewhere. Higher
x86 ISAs and SVE require an explicit `LINALG_CPU_BACKEND` and a CPU that
supports it; there is no runtime ISA dispatch. Universal macOS builds require
`LINALG_CPU_BACKEND=no`. ISA flags stay private to the implementation.
LTO modes and linker names are validated at configure time. CUDA builds require
LTO off. Explicit per-target linkers require a GNU compiler driver; MSVC and
clang-cl use the linker selected by their toolchain file. Sanitizer and coverage runtimes propagate to static-library consumers;
instrumentation applies to LinearAlgebra, while dependencies use their own options.
Windows Clang ASan builds disable STL string/vector container annotations to
match uninstrumented dependencies; ordinary ASan memory checks remain enabled.
MSVC debug information is embedded in object files (`/Z7`) so parallel compiler
cache invocations do not share a compiler PDB.
Valgrind checks run through `ctest -T memcheck` when `LINALG_ENABLE_VALGRIND=ON`.

From `Scripts/`, use `python3 setup.py config.build.test.hipblas` to enable
the hipBLAS/hipSOLVER backend (`LINALG_ENABLE_ROCBLAS=ON`). This requires a
ROCm/HIP toolchain with hipBLAS and hipSOLVER installed.

On macOS, use `python3 setup.py config.build.test.metalblas` from `Scripts/`
to enable Metal Performance Shaders (`LINALG_ENABLE_METAL=ON`). MetalBLAS
supports float multiplication, batched multiplication, transpose, norm, and
trace. The configuration summary reports `MetalBLAS: ON` when enabled.
The `cublas`, `hipblas`, and `metalblas` tokens are mutually exclusive;
running the Metal GPU tests requires a Metal-capable host.

Icecream can be combined with `ccache` or `sccache`, for example with
`python3 setup.py config.build.test.ICECC.SCCACHE` from `Scripts/`.
Cache hits reuse existing objects; cache misses compile through Icecream.
The generated compiler wrappers fingerprint the real compiler and Icecream
executables so tool updates invalidate cached objects.

With both CPU vendor backends enabled, `blas_lapacke` calls its own shared
provider and `mkl` calls MKL's ILP64 interface. The BLAS provider is loaded
through a private handle rather than linked alongside MKL, since the libraries
export overlapping CBLAS/LAPACKE symbols. Its headers must match its integer ABI.
For example, on Windows:

```powershell
cmake -G Ninja -S . -B build_compare -DCMAKE_BUILD_TYPE=Release -DLINALG_ENABLE_MKL=ON -DLINALG_ENABLE_BLAS=ON -DLINALG_ENABLE_BENCHMARK=ON -DLINALG_CBLAS_INCLUDE_DIR=C:/tools/OpenBLAS/include -DLINALG_LAPACKE_INCLUDE_DIR=C:/tools/OpenBLAS/include -DLINALG_BLAS_RUNTIME_LIBRARY=C:/tools/OpenBLAS/bin/libopenblas.dll
cmake --build build_compare --config Release
./build_compare/bin/benchmark_linalg_multiplication.exe --benchmark_filter='double.*(BLAS_LAPACKE|MKL)>/64/64/64'
```

Each benchmark executable reports both backends separately. Match vendor thread
counts (for example `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`) when comparing
them. Function addresses are cached on first use; combined builds add an indirect
call to BLAS operations. Standalone BLAS builds retain direct calls.

Eigen reference tests are always included in the test suite. To build the
benchmark comparisons as well:

```sh
cmake -S . -B build_eigen -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLINALG_ENABLE_TESTING=ON \
  -DLINALG_ENABLE_BENCHMARK=ON
cmake --build build_eigen --parallel 4
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  ctest --test-dir build_eigen -R '^LinearAlgebraCxxTests$' --output-on-failure
./build_eigen/bin/LinearAlgebraCxxTests --gtest_filter='EigenReference*'
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  ./build_eigen/bin/benchmark_linalg_multiplication \
  --benchmark_filter='double.*64/64/64/' --benchmark_out=eigen-matmul.json \
  --benchmark_out_format=json
```

Eigen 3.4.0 is pinned as the `ThirdParty/eigen` Git submodule. Initialize
third-party dependencies with `git submodule update --init --recursive` before
building with CMake or Bazel. Eigen comparisons are always active in every
built test and benchmark target; no Eigen feature flag is required. Eigen stays
private to these targets, so the installed library has no Eigen dependency.
Enable BLAS or MKL as above to include those CPU backends in the comparisons.

The float/double reference tests check multiplication with transposes and padded
strides, LU factors and reused solves, both Cholesky triangles, QR and SVD
reconstruction and orthogonality, symmetric eigenpairs, and real/complex general
spectra against every enabled CPU backend. Comparisons allow factor signs and
spectrum ordering to differ.

The benchmark programs add `BM_Eigen*` rows using the same
seed, row-major inputs, dimensions, and requested outputs as their library rows.
Eigen runs single-threaded. LU uses Eigen partial pivoting even when native LU
pivoting is disabled. QR includes explicit Q formation; SVD uses Eigen JacobiSVD
with thin U and explicitly transposed VT. Input refresh is excluded from timing
except for SVD, matching the existing benchmarks. Decomposition objects and output buffers are reused;
internal Eigen work and copies remain timed. Run Release builds and compare the
matching operation, scalar type, size, and output mode.

`benchmark_linalg_schurdecomposition` compares full complex Schur form and
unitary vectors on dense nonsymmetric real inputs, checking reconstruction,
unitarity, and triangular form before timing. Eigen's real-to-complex input
conversion and split real/imaginary output copies are included in timing.
The native, BLAS, and MKL Schur backends currently share the portable algorithm.
General eigenpair benchmarks also include every enabled CPU backend, using the
same real-spectrum inputs as Eigen.

The additional `benchmark_linalg_kernels` executable compares in-place transpose,
Frobenius/one/infinity norms, LU factorization and solve, solves with existing LU
factors, inversion, and lower/upper triangular solves with and without transpose.
Triangular cases include one and sixteen right-hand sides. All CPU backends
compiled into the build are compared with Eigen on the same inputs. The norm
implementations are shared by the CPU backends. Upfront LU is excluded from
factorized solves; other solves and inversion include factorization.

Native multiplication writes directly into row-major output using the transposed
product identity, avoiding an output transpose and rectangular output scratch.
Native symmetric eigenpairs use Householder tridiagonalization and implicit QL;
general eigenvalues use the portable Hessenberg/shifted Schur iteration, skipping
Schur-vector accumulation when only values are requested. Real eigenvectors use
triangular back substitution and the Schur vectors. Native complex eigenvectors
remain unsupported. Square transpose uses cache tiles with zero workspace, while
rectangular transpose uses the existing permutation cycles. Cholesky packs short
transposed panels for its SIMD trailing update. CPU Frobenius and infinity norms
use independent partial sums; the one norm traverses short contiguous column
blocks to improve cache use. These norm changes apply to all CPU backends.

The [recorded Eigen comparison](Testing/Benchmark/results/2026-10-23/REPORT.md)
includes test results, timings for every registered case, CSV data, scaling charts,
and hardware/build metadata. The [optimized comparison](Testing/Benchmark/results/2026-10-23-optimized/REPORT.md)
adds native before/after measurements and the additional kernel workloads.
Published results retain final CSVs, charts, and metadata; raw runs, logs, and
intermediate measurements are generated locally and ignored by Git.
To produce a fresh comparison from a Release build:

```sh
python3 Testing/Benchmark/run_eigen_comparison.py --build-dir build_eigen \
  --output-dir /tmp/eigen-comparison
python3 Testing/Benchmark/summarize_eigen_comparison.py /tmp/eigen-comparison
```

Use `--operations kernels eigen` to select executables or `--filter 'NATIVE.*>/128/'`
to select case names. Choose a fresh output directory for each run. To compare
native speedups against a previous run, pass `--baseline /path/to/previous-results`
to the summarizer; `--no-charts` omits the Matplotlib dependency. MKL and
BLAS/LAPACKE results retain separate columns.

For LU, Schur, and eigenvalues/eigenvectors only, use
`--operations ludecomposition schurdecomposition eigen`.
The [recorded LU, Schur, and eigensolver comparison](Testing/Benchmark/results/2026-10-24-lu-schur-eigen/REPORT.md)
includes 162 float/double cases across native, OpenBLAS, and Eigen.

The runner executes the unit tests first, then benchmarks each case sequentially
with one thread per backend. It preserves timeouts and failures alongside raw
Google Benchmark JSON and logs. The report script requires Matplotlib.

The [CPU/GPU benchmark example](examples/backend_benchmark/README.md) compares
all four backends in one executable. Its [Markdown analysis](examples/backend_benchmark/analysis.md)
includes recorded timings, transfer costs, workspace sizes and scaling charts.

The configured shared-library paths and their dependencies must remain available
at runtime, including when consuming an installed LinearAlgebra package. Set
`LINALG_LAPACKE_RUNTIME_LIBRARY` explicitly if LAPACKE is packaged separately
(for example `liblapacke.so`). On ELF systems, combined builds require
`RTLD_DEEPBIND` to isolate the provider's internal calls; sanitizer checks must
enable one CPU vendor backend at a time.

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
| **cuBLAS / cuSOLVER** | NVIDIA GPU | ✅ | ✅ | ❌ | ❌ | ✅ `LINALG_ENABLE_CUBLAS` (`linalg_backend::CUDA_BLAS`) |
| **BLIS** | CPU | ✅ | ❌ | ❌ | ❌ | ✅ via `LINALG_ENABLE_BLIS`; pair with LAPACKE for decompositions |
| **cuBLASLt** | NVIDIA GPU | ✅ | — | — | — | ❌ not wired up |
| **cuSPARSE** | NVIDIA GPU | — | — | ✅ | — | ❌ out of scope (dense only) |
| **hipBLAS / hipSOLVER** | AMD GPU | ✅ | ✅ | ❌ | ❌ | ✅ `LINALG_ENABLE_ROCBLAS` (`linalg_backend::ROCM_BLAS`) |
| **rocSPARSE** | AMD GPU | — | — | ✅ | — | ❌ out of scope (dense only) |

\* OpenBLAS ships LAPACK only when built with its bundled reference LAPACK
(the common case for prebuilt packages); a BLAS-only OpenBLAS build falls
back to the scalar path here for LAPACK-level ops (see `LINALG_ENABLE_BLAS`
above). FFT and sparse routines are out of scope for LinearAlgebra, which is
dense-only; the FFT/Sparse columns above describe each vendor library in
general, not something this repo calls. `LINALG_ENABLE_CUBLAS` and
`LINALG_ENABLE_ROCBLAS` are mutually exclusive — enable only the one
matching the target GPU vendor (see `docs/gpu_backend_design.md`).

## Public API (abridged)

```cpp
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver.h"

// In-place lower-triangular Cholesky factor of an n x n row-major matrix.
// Choose NATIVE, BLAS_LAPACKE, or MKL explicitly for each operation.
linalg::cholesky_decomposition<double, linalg::linalg_backend::NATIVE>::apply(
    A.data(), n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);

// Solve A x = b in place, reusing the Cholesky factor above.
linalg::linear_solver<double, linalg::linalg_backend::NATIVE>::apply(
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
`-DLINALG_ENABLE_BLAS=ON`, the NVIDIA GPU backend with
`-DLINALG_ENABLE_CUBLAS=ON` (requires the CUDA toolkit; set
`CMAKE_CUDA_ARCHITECTURES` for your GPU, or leave it to `native` detection),
or the AMD GPU backend with `-DLINALG_ENABLE_ROCBLAS=ON` (requires a
ROCm/HIP toolchain; set `CMAKE_HIP_ARCHITECTURES` for your GPU). The two GPU
backends are mutually exclusive.

C++-only hosts can use `add_subdirectory`; bundled Logging's C language needs
are enabled automatically. Existing `Logging::Logging` and `Scalars::Scalars` targets are reused.
Set `LINALG_USE_SYSTEM_LOGGING=ON` or `LINALG_USE_SYSTEM_SCALARS=ON` to require
the respective installed config package.

Installation uses `CMAKE_INSTALL_LIBDIR`, `CMAKE_INSTALL_BINDIR`, and
`CMAKE_INSTALL_INCLUDEDIR`. Scalars is installed with LinearAlgebra when bundled.
Logging is a separate package: install it first, then install LinearAlgebra
and put both prefixes on the consumer's
`CMAKE_PREFIX_PATH`. Custom library directories may require an explicit
`LinearAlgebra_DIR=<prefix>/<libdir>/cmake/LinearAlgebra` when consuming the package.

Run the configuration regressions with:

```bash
python3 Testing/CMake/check_configurations.py
```

## Bazel

```bash
bazel test //...
```

Bazel version is pinned via [`.bazelversion`](.bazelversion): `WORKSPACE.bazel`
wires `ThirdParty/googletest` in via `local_repository`.

## CI & Coverage

[`.github/workflows/flag-combinations.yml`](.github/workflows/flag-combinations.yml)
adds constrained pairwise flag coverage on pushes and pull requests to `main`:
Linux GCC/Clang, macOS ARM64 Clang, and Windows MSVC/Clang. The deterministic
plan currently builds 176 configurations in 24 jobs, covering every feasible
pair of values across 24 build dimensions on each toolchain. Dedicated cases
activate Icecream together with each of ccache and sccache. Each case configures,
checks effective target properties, builds, and runs enabled CPU tests; requested
flags, effective settings, results, and full logs are uploaded as artifacts.
Shared/static builds, four build types, C++20/23, Ninja single/multi-config,
testing switches, OpenBLAS/BLIS, SIMD, caches, LTO, linkers, instrumentation,
benchmarks, and bundled/installed Logging are included.

Use **Actions → Flag combinations → Run workflow** for manual exhaustive windows:
choose `mode=exhaustive`, a toolchain (or `all`), a Cartesian `offset`, and a
`window` of 1–2048 assignments per toolchain. The uploaded `flag-plan.json`
records the total space, excluded assignments, and `next_offset`; dispatch the
next window to continue. One dispatch covers only its selected window. The
exhaustive space uses the same build dimensions as the pairwise plan; optional
vendor backends and developer tools retain the dedicated checks below.
For a local preview, run:

```sh
python3 .github/scripts/flag_matrix.py --output build/flag-plan.json
python3 -m unittest discover -s Testing/CMake -p test_flag_matrix.py -v
```

Platform domains exclude unavailable options: Icecream and BLIS on hosted
Windows, GCC ThinLTO, native MSVC sanitizers, and unsupported sanitizer/ISA
values. Effective LTO with sccache, Icecream with buildcache, and GCC LTO with
LLVM's linker are excluded. Instrumentation disables LTO as the project
requires. Unsafe ISA execution and MemorySanitizer/ThreadSanitizer runtime
checks are recorded as compile-only; these builds still run the source check
when testing is enabled. Analyzer checks exercise all 16 combinations of
clang-tidy/fix/IWYU/spelling switches with recording tools. Profiling checks
verify dependency/target contracts; neither suite claims real analyzer or
Profiler execution. Configuration regressions run on all five toolchains,
and the option inventory fails when a new project switch lacks a coverage owner.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
PR to `main`: native CMake builds with GCC and Clang, OpenBLAS/LAPACKE and
MKL builds, LU pivoting, sanitizers, and `bazel test //Testing/...`. MKL runs on
Linux and Windows with shared/static libraries and with/without OpenBLAS.
The SIMD matrix covers vectorization disabled and each explicit ISA; an ISA
is compiled but not executed when the runner lacks the required instructions.
Shared and static install-consumer jobs build a separate application against
the installed packages. Coverage runs on Linux/macOS/Windows for the native
backend and on Linux for OpenBLAS/LAPACKE and MKL, with separate report artifacts
and Codecov flags per backend.

CI fetches pinned recursive submodules in a separate step with up to five
attempts and delays of 15, 30, 60, and 120 seconds. This lets temporary
dependency-hosting failures recover; checkout still fails if all attempts fail.

CUDA is compiled without a GPU on Linux and Windows. Linux additionally
compiles CUDA with OpenBLAS, MKL, and the example requiring all three backends.
GPU runtime tests run on pushes
to `main` when the repository variable `LINALG_GPU_CI_ENABLED` is `true` and a
self-hosted runner with labels `linux`, `x64`, and `gpu` is registered. That runner
needs Docker, a compatible NVIDIA driver, and the NVIDIA Container Toolkit;
the job installs its build tools inside the CUDA container. Until that runner
is available, GPU numerical correctness requires manual hardware testing.

HIP is compiled without a GPU on Linux the same way, inside a
`rocm/dev-ubuntu-22.04` container, across the same OpenBLAS/MKL/examples
matrix (`LINALG_ENABLE_ROCBLAS=ON` in place of `LINALG_ENABLE_CUBLAS=ON`;
the two are mutually exclusive, so this is a separate `hip-compile` job
rather than another matrix row on `cuda-compile`). There is no HIP Windows
compile job: unlike CUDA, there is no maintained GitHub Action for
installing the ROCm HIP SDK on Windows runners. HIP GPU runtime tests run
the same way as CUDA's, gated behind the repository variable
`LINALG_ROCM_GPU_CI_ENABLED` and a self-hosted runner with labels `linux`,
`x64`, and `rocm`; that runner needs Docker, the `amdgpu` kernel driver, and
`/dev/kfd`/`/dev/dri` exposed to the container. Until that runner is
available, HIP numerical correctness likewise requires manual hardware testing.
See [CONTRIBUTING.md](CONTRIBUTING.md)
for running the same checks locally, plus sanitizers and lintrunner (not yet
wired into CI).
