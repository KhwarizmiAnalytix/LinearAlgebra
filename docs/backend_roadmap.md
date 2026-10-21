# LinearAlgebra design review and backend roadmap

Status: proposal. Nothing in this document is implemented yet. Phases in §6
are ordered by dependency; each one is independently buildable and testable
and must leave CI green before the next starts.

Scope: the whole library — public API, CPU backend selection, the GPU tree,
build systems, and CI. [`gpu_backend_design.md`](gpu_backend_design.md)
remains the detailed design for the GPU backend; where this roadmap touches
GPU work (§6, Phase 4) it extends that document's §5.4 rather than replacing
it.

**Memory ownership principle.** CPU and GPU memory management belongs to
the [KhwarizmiAnalytix/Memory](https://github.com/KhwarizmiAnalytix/Memory)
repository (unified `allocator<T>`, CUDA/HIP/Metal caching allocators,
mimalloc/TBB host allocators, `data_ptr`/`data_view` ownership types), not
to LinearAlgebra. LinearAlgebra is a pure kernel library: it computes on
buffers it is handed and never allocates, caches, or frees memory —
including internal scratch. Callers obtain every buffer, workspace
included, from the Memory library (or any allocator of their choosing) and
pass it in as a raw pointer. LinearAlgebra takes no build-time dependency
on the Memory repository; the contract between the two is "raw pointer plus
size", which keeps both libraries independently consumable.

## 1. Current architecture

| Layer | Today |
|---|---|
| Public API | Free functions in `linalg::`, row-major raw `T*` + leading dimension, no matrix type, caller owns all memory. GPU counterparts live in `linalg::gpu::*` (device pointers, optional trailing `cudaStream_t`). |
| CPU backend selection | Compile-time `#if LINALG_ENABLE_MKL / #elif LINALG_ENABLE_BLAS [&& LINALG_BLAS_HAS_LAPACKE] / #else` scalar chain, copied into each op's `.cpp`. Replaced a PyTorch-style `dispatch_stub` registry in commit `b09f66e`. |
| CPU backends | Scalar fallback; Intel MKL (implicitly ILP64); vendor-neutral CBLAS/LAPACKE (OpenBLAS, netlib, ...). Apple Accelerate is reached only through `LINALG_ENABLE_BLAS` and covers GEMM/transpose only, because it ships no LAPACKE. |
| GPU backend | cuBLAS/cuSOLVER for every CPU op except general (non-symmetric) eigenvalue decomposition. Per-thread, per-device handle pool with stream binding ([`cuda_handle.h`](../include/common/cuda_handle.h)). |
| Memory | Callers own inputs and outputs, but the library still allocates internal scratch itself: `cudaMalloc`/`cudaFree` in 13 GPU sources, [`linalg::allocator`](../include/memory/allocator.h) in 9 CPU sources, and `std::vector` temporaries in most CPU ops. This conflicts with the ownership principle above. |
| Composite ops | `least_squares`, `pseudo_inverse`, `rank`, `norm`, `trace` build on SVD and inherit its backend. |
| Outside the backend model | Tri-/penta-/multi-diagonal operations, `exponential`, `cholesky_decomposition_aad` — scalar only, with their own API conventions. |
| Build systems | CMake is primary. Bazel covers the CPU tree, with an MKL toggle but no BLAS toggle. |
| CI | Scalar (Linux, macOS), MKL (Linux), Bazel (Linux, macOS), coverage (Linux, macOS, Windows). Not covered: BLAS/LAPACKE, `LINALG_LU_PIVOTING`, CUDA, sanitizers. |

Strengths worth preserving:

- The LAPACK-style contract for inputs and outputs: the caller allocates
  and owns them.
- Zero per-call dispatch cost on the CPU path.
- A GPU API that keeps data device-resident and accepts caller streams.
- Two build systems and a real vendor-backend CI job (MKL).
- Design decisions recorded in-tree (`gpu_backend_design.md`).

## 2. Findings

### 2.1 Correctness and robustness

1. **Default scalar LU does not pivot.** `LINALG_LU_PIVOTING` defaults to
   `OFF`, so the default build is numerically unsafe and returns different
   results than MKL/BLAS for the same call. The pivot buffer contract also
   changes with the flag: the scalar pivoting path writes `pivot[lda]` and
   needs `n + 1` entries. Semantics that depend on build flags are the
   hardest kind of backend divergence to diagnose.
2. **Index types.**
   - `linalg_int` / `linalg_long` are preprocessor macros redefined in every
     public header (for example
     [`multiplication.h`](../include/matrix_operation/multiplication.h)).
     They leak into the global namespace, and signed (`linalg_int`) and
     unsigned (`linalg_long`, used by SVD) indices are mixed.
   - The MKL path passes `linalg_int*` directly as `MKL_INT*`, which only
     works with an ILP64 MKL. CMake never pins `MKL_INTERFACE`, and no
     `static_assert` guards the assumption.
   - The BLAS/LAPACKE path and every GPU op narrow with
     `static_cast<int>(...)` without checking for overflow (for example
     [`linear_solver_gpu.cpp`](../include/matrix_operation_gpu/linear_solver_gpu.cpp)).
3. **The library allocates internally, against the ownership principle.**
   - GPU: 13 sources call `cudaMalloc`/`cudaFree` for workspace, pivot
     arrays, and intermediate buffers. Because plain `cudaFree` is not
     stream-ordered, each call goes through `detail::synchronize_and_free`,
     i.e. `cudaMalloc` → work → `cudaStreamSynchronize` → `cudaFree`. The
     headers state the calls "never block on a device synchronization"; in
     practice every factorization is synchronous, which undermines the
     stream parameter. It also bypasses the Memory library's caching
     allocator, so a caller looping over solves pays a raw device
     allocation per call.
   - CPU: 9 sources allocate scratch through
     [`linalg::allocator`](../include/memory/allocator.h), and most ops
     build `std::vector` temporaries (LAPACKE pivot conversion, SVD/QR/eigen
     work arrays). None of it is visible to, or controllable by, the
     caller's allocator.

### 2.2 API shape

4. Square ops conflate `n` and `lda` (`lu_decomposition(m, lda, pivot)`),
   so padded storage and sub-block factorization are impossible.
   `linear_solver` accepts a single right-hand side only (CPU and GPU).
5. The diagonal operations follow a different convention from the rest of
   the library:
   - `std::vector<double>&` instead of raw pointers;
   - `double` only;
   - `size_t` indices;
   - a `parallelize` parameter that is silently ignored
     (`(void)parallelize;`).

### 2.3 Maintainability

6. The three-way backend `#if` chain is duplicated across about nine
   translation units; adding a backend means editing every one of them.
7. About 1,500 lines of Eigen-derived blocked GEMM in
   [`multiplication.cpp`](../include/matrix_operation/multiplication.cpp)
   sit behind `LINALG_VECTORIZED`, which is never defined and includes a
   `common/packet.h` that does not exist. The live scalar GEMM is a naive
   loop, and [`cpu_info.h`](../include/util/cpu_info.h) is a stub.
8. Stale documentation:
   - [`svd_decomposition.h`](../include/matrix_operation/svd_decomposition.h),
     [`eigen.h`](../include/matrix_operation/eigen.h)
     and [`inversion.h`](../include/matrix_operation/inversion.h)
     state that no GPU implementation exists; one does.
   - [`BUILD.bazel`](../BUILD.bazel) justifies `alwayslink = True` with the
     removed `dispatch_stub` registrars.
   - The header comment of [`CMakeLists.txt`](../CMakeLists.txt) lists only
     Cholesky/LU/SVD.
   - The README module table omits `exponential` and the diagonal
     operations.
9. No benchmarks exist, so performance-dependent decisions (for example the
   batched-GEMM size cutoff in `gpu_backend_design.md` §9) cannot be made
   from data.
10. Backends cannot be compared against each other: exactly one CPU backend
    is compiled per binary, so the scalar code cannot act as a reference
    when testing MKL or BLAS.

## 3. Backend landscape

| Backend | Status | Value | Effort | Notes |
|---|---|---|---|---|
| Scalar | Shipped | Reference, portable | — | Naive GEMM; non-pivoting LU by default (§2.1). |
| Intel MKL | Shipped, CI | High on x86 | Low | Pin `MKL_INTERFACE=ilp64`; add `static_assert(sizeof(MKL_INT) == sizeof(index_t))`. |
| CBLAS/LAPACKE (OpenBLAS, netlib, BLIS, AMD AOCL) | Shipped, no CI | High | Low | Mostly a CI matrix entry. AOCL provides LAPACKE through libflame. |
| Apple Accelerate, full LAPACK | GEMM/transpose only | High on macOS | Low–medium | macOS 13.3+ ships LAPACK 3.9 behind `ACCELERATE_NEW_LAPACK` (+ `ACCELERATE_LAPACK_ILP64`). Column-major Fortran-style entry points, no LAPACKE, so it needs the transpose/relabel bookkeeping the GPU ops already use. |
| cuBLAS/cuSOLVER | Shipped, no CI | High | Medium | Move to the 64-bit `cusolverDnX*` / `cublas*_64` APIs. Newer cuSOLVER releases (CUDA 12.6 Update 2 onward) add `cusolverDnXgeev`, which may close the one GPU coverage gap — verify against the targeted toolkit before relying on it. |
| Eigen3 (optional CPU backend) | Not present | Medium–high | Medium | Header-only, MPL-2.0 (compatible with this repository's licensing). Supplies a vectorized GEMM, pivoted LU, BDCSVD and eigen-solvers without a system BLAS, and supersedes the dead GEMM code from §2.3. |
| ROCm (hipBLAS/hipSOLVER) | Shipped, no CI | Medium | Medium–high | hipSOLVER mirrors cuSOLVER closely; PyTorch reaches ROCm by hipifying its CUDA sources the same way. `linalg_backend::ROCM_BLAS` / `LINALG_ENABLE_ROCBLAS`, mutually exclusive with `LINALG_ENABLE_CUBLAS` (both would define the same non-backend-templated `batched_matrix_multiplication` symbols). |
| MAGMA | Not present (non-goal in `gpu_backend_design.md` §2) | Low–medium | Medium | Hybrid CPU/GPU `geev`. Only worth it if `cusolverDnXgeev` is unusable on the targeted toolkits. |
| Apple MPS (Metal Performance Shaders) | Shipped, no CI | Medium on Apple platforms | — (shipped) | `float`-only (no native Metal `double`), and covers only multiplication / batched multiplication / transpose / norm / trace — MPS has no QR/SVD/general-solver routines, so cholesky/LU/QR/SVD/eigen/linear_solver/inversion/determinant/rank/pseudo_inverse/least_squares have no `METAL_BLAS` specialization (same shape as the general-eigenvalue gap row 134 describes for CUDA/HIP). `linalg_backend::METAL_BLAS` / `LINALG_ENABLE_METAL`, mutually exclusive with `LINALG_ENABLE_CUBLAS`/`LINALG_ENABLE_ROCBLAS` (all three define the same non-backend-templated `batched_matrix_multiplication` symbols). Pointers must be page-aligned (wrapped zero-copy via `newBufferWithBytesNoCopy:`, see `include/common/metal_handle.h`). |
| oneMKL SYCL | Not present | Low | High | Small audience here. Deferred. |

## 4. Architectural direction

Keep backend selection at compile time — the `b09f66e` decision against a
runtime registry stands — but restructure how backends plug in.

**Reference points.** Eigen3 keeps its native implementation always
present and layers vendor specializations on top per op
(`EIGEN_USE_BLAS` / `EIGEN_USE_LAPACKE`, for example
`Eigen/src/Core/products/GeneralMatrixMatrix_BLAS.h`). PyTorch wraps LAPACK
in typed templates (`lapackLu<T>`, `lapackCholesky<T>` in
`aten/src/ATen/native/BatchLinearAlgebra.cpp`) so op code never sees a
vendor's raw symbol names, and selects the vendor at build time
(`AT_MKL_ENABLED()`, `AT_BUILD_WITH_LAPACK()`). This roadmap borrows the
first pattern for structure and the second for the binding layer; it does
not adopt PyTorch's runtime `linalgPreferredBackend()` switch.

1. **One internal LAPACK binding layer**,
   `include/common/detail/lapack.h`: typed overloads such as
   `detail::lapack::getrf(float*, ...)` / `getrf(double*, ...)`. The
   MKL / LAPACKE / Accelerate selection lives in that one header (plus its
   `src/common/detail/lapack.cpp`), and each op's `.cpp` reduces to
   `#if LINALG_HAS_LAPACK ... #else reference ... #endif`. Adding a backend
   becomes a one-file change.
2. **Scalar code always compiled as `detail::reference`.** The public entry
   point still binds to exactly one backend at compile time, but tests can
   call the reference directly and compare it with the active backend in
   the same binary. Cost: binary size, no runtime dispatch.
3. **Workspace query plus caller-supplied workspace**, following LAPACK's
   `lwork = -1` query and cuSOLVER's `*_bufferSize`. Every op that needs
   scratch exposes a `*_workspace_size(...)` query (bytes, plus required
   alignment) and an overload taking `void* workspace, std::size_t
   workspace_bytes`. The caller obtains that buffer from the Memory library
   — host or device, cached or not — and LinearAlgebra carves pivot arrays
   and work arrays out of it. The library itself never calls `cudaMalloc`,
   `cudaMallocAsync`, `operator new`, or `std::vector` for scratch. This
   removes the forced GPU synchronization (§2.1, item 3) because nothing
   is freed inside the call, and it lets the Memory library's caching
   allocator amortize workspace across calls. PyTorch makes the same split:
   ATen's linalg kernels request cuSOLVER workspace from
   `CUDACachingAllocator` instead of allocating it themselves; here the
   allocator lives in a separate repository and the hand-off is an explicit
   parameter rather than a global allocator lookup.
4. **Introspection only.** `Context::cpu_backend()` reports which backend
   was compiled in. It is not a runtime switch.

## 5. Non-goals

- No runtime backend registry or per-call backend override on the CPU path.
- No memory management of any kind in LinearAlgebra: no allocator, no
  caching pool, no owning buffer or matrix/tensor type, no stream-ordered
  allocation fallback. That is the Memory repository's responsibility;
  raw `T*` plus caller-supplied workspace stays the public contract.
- No build-time dependency on the Memory repository. Tests and benchmarks
  may use it, or a test-only helper, to provision buffers.
- No complex or half-precision types in this roadmap.
- No change to the CMake-only status of the GPU tree under Bazel.

## 6. Phased plan

### Phase 0 — Hygiene, no behavior change

- [ ] Replace the per-header `linalg_int` / `linalg_long` macros with a
      single `include/common/types.h` (`using index_t = std::int64_t;`).
      Keep the old macros as deprecated aliases until 2.0.
- [ ] Add `detail::checked_narrow<To>(value)` that reports overflow through
      `LOGGING_THROW`, and use it at every `static_cast<int>` on the
      BLAS/LAPACKE and GPU paths.
- [ ] Pin `MKL_INTERFACE=ilp64` in `CMakeLists.txt` and add the `MKL_INT`
      width `static_assert`.
- [ ] Delete the dead `LINALG_VECTORIZED` GEMM and the `cpu_info.h` stub;
      Phase 5's Eigen3 backend covers the performance need.
- [ ] Fix the stale comments listed in §2.3 item 8 and complete the README
      module table.

Exit criteria: CI green; diff contains no numerical changes.

### Phase 1 — Test and CI infrastructure

- [ ] CI jobs:
  - OpenBLAS + LAPACKE (Linux via apt, macOS via Homebrew keg);
  - `LINALG_LU_PIVOTING=ON`;
  - AddressSanitizer + UndefinedBehaviorSanitizer;
  - CUDA compile-only build in an `nvidia/cuda` container (catches GPU-tree
    build breaks without a GPU runner).
- [ ] Bazel: add a `linalg_enable_blas` `config_setting` next to the
      existing MKL one.
- [ ] Always compile `detail::reference` and add a parity suite under
      `Testing/Cxx/` comparing each op's active backend with the reference
      within per-type tolerances. Register it in both CMake and Bazel.
- [ ] Add `Testing/Benchmark/` using Google Benchmark (already vendored
      inside `ThirdParty/Logging`): GEMM, LU, Cholesky, SVD over a size
      sweep, CPU and GPU.
- [ ] Add a `find_package(LinearAlgebra)` consumer test to CI to cover the
      install/export path.

Exit criteria: every backend runnable in CI is exercised, and any
disagreement between a backend and the reference fails a test.

### Phase 2 — CPU backend consolidation

- [ ] Introduce the LAPACK binding layer (§4, item 1) and port each op's
      `#if` chain onto it, one op per pull request.
- [ ] Add full-LAPACK Accelerate support through the binding layer, reusing
      the row-major ↔ column-major transpose/relabel approach proven in the
      GPU ops. Confirmed by compiling a probe against this build's actual
      flags (`LINALG_HAS_ACCELERATE=1, LINALG_HAS_BLAS=1,
      LINALG_HAS_LAPACKE=0`):
  - GEMM/transpose already reach Accelerate today, with no further work:
    [`multiplication.h`](../include/matrix_operation/multiplication.h) and
    [`transpose.h`](../include/matrix_operation/transpose.h) each define
    their own `kMatmulActiveBackend`/`kTransposeActiveBackend`, gated on
    `LINALG_HAS_BLAS` alone, independent of the global `kActiveBackend`
    ([`include/common/backend.h`](../include/common/backend.h)) that every
    other op uses.
  - Cholesky/LU/SVD/inversion/linear_solver stay on `native` on an
    Accelerate-only macOS build because `kActiveBackend` requires
    `LINALG_HAS_BLAS && LINALG_HAS_LAPACKE`, and each op's
    `src/matrix_operation/blas/*.cpp` wraps its *entire* body in
    `#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE` — when LAPACKE is absent the
    `blas_lapacke` specialization isn't compiled at all, so there is no
    existing fallback body to repurpose; the classic-Fortran-symbol path
    needs new code per op, following the `kMatmulActiveBackend` precedent
    (a per-op backend constant, not a change to the shared `kActiveBackend`).
  - Until this lands, the supported way to get `blas_lapacke` for these five
    ops on macOS is unchanged: install a LAPACKE-providing package (for
    example `brew install openblas`). `CMakeLists.txt`'s `LINALG_ENABLE_BLAS`
    block now probes `brew --prefix openblas` itself and appends the result
    to `CMAKE_PREFIX_PATH` on Apple platforms, so `find_path(... lapacke.h)`
    succeeds without a manual `-DCMAKE_PREFIX_PATH` — only a non-Homebrew
    LAPACKE install still needs that passed explicitly, per the
    `FATAL_ERROR` message / the comment at `CMakeLists.txt`:344-350.
- [ ] Make scalar LU pivot by default: flip `LINALG_LU_PIVOTING` to `ON`,
      deprecate `OFF`, and standardize the pivot buffer on `n` entries,
      1-based, matching LAPACK.
- [ ] Add `Context::cpu_backend()`.

Exit criteria: adding a backend touches one file; Accelerate covers every
op on macOS; the parity suite passes for scalar, BLAS/LAPACKE, MKL, and
Accelerate.

### Phase 3 — API 1.1 (additive only)

- [ ] General overloads taking `n` and `lda` separately (for example
      `lu_decomposition(n, m, lda, pivot)`), keeping the square shorthands.
- [ ] Multi-right-hand-side solve (`nrhs`, `ldx`) on CPU and GPU — the
      first half of `gpu_backend_design.md` §5.4.
- [ ] `*_workspace_size()` queries and caller-workspace overloads for every
      CPU op that needs scratch (§4, item 3), replacing `linalg::allocator`
      and `std::vector` temporaries inside the workspace overloads. The
      existing no-workspace overloads stay for 1.x, marked deprecated, as
      the only remaining internal-allocation path.
- [ ] Align the diagonal operations with the library convention:
      `float`/`double`, raw pointers, `index_t`; implement or remove
      `parallelize`.
- [ ] Evaluate a `linalg::status` return (LAPACK-style info code) in place
      of `bool` for the new overloads.

Exit criteria: every existing call site compiles unchanged; new overloads
are covered by the parity suite on every backend.

### Phase 4 — GPU completion

- [ ] Migrate to the 64-bit `cusolverDnX*` / `cublas*_64` APIs.
- [ ] `*_workspace_size()` queries and caller-workspace overloads for every
      GPU op, covering cuSOLVER workspace, device pivot arrays, and
      intermediate buffers (SVD/QR/inversion/pseudo-inverse). Remove every
      `cudaMalloc`/`cudaFree` and `detail::synchronize_and_free` from the
      workspace path, so non-reducing ops no longer synchronize. The
      existing no-workspace overloads stay for 1.x, deprecated.
- [ ] Add a test that captures a solve in a CUDA graph and replays it with
      a workspace provisioned by the Memory library's CUDA allocator.
- [ ] Batched factorizations and `cublasXgemmStridedBatched`
      (`gpu_backend_design.md` §5.4), with the size cutoff taken from
      Phase 1 benchmark data.
- [ ] `eigen` on the GPU via `cusolverDnXgeev`, guarded
      by a CUDA toolkit version check.
- [ ] Update `gpu_backend_design.md` accordingly. Its §2 currently keeps
      per-call `cudaMalloc`/`cudaFree` workspace "by design"; that must be
      replaced by the caller-workspace contract and the Memory-repository
      ownership principle from this document. Revise §8 and §9 to match.

Exit criteria: full CPU/GPU op parity, and the documented non-blocking
contract matches what the code does.

### Phase 5 — New backends (demand-driven)

- [ ] Optional Eigen3 CPU backend (`LINALG_ENABLE_EIGEN`), vendored under
      `ThirdParty/` per [the dependency rules](../.augment/rules/ThirdParty.md).
      Define precedence relative to MKL and BLAS explicitly.
- [x] ROCm through hipBLAS/hipSOLVER, shipped as `linalg_backend::ROCM_BLAS`
      / `LINALG_ENABLE_ROCBLAS` under `src/matrix_operation/gpu/hipblas/`,
      with a parallel `hip_handle.h`/`hip_fwd.h`/`hip_workspace.h` rather
      than generalizing `cuda_handle.h` itself (kept the CUDA tree
      untouched; `linalg_backend` already provides the vendor-neutral
      dispatch point `detail::transpose_device`-style free functions would
      have duplicated). CI-exercised compile-only on every push/PR
      (`hip-compile`, no ROCm runner needed); GPU runtime is gated behind a
      self-hosted ROCm runner (`hip-runtime`, `LINALG_ROCM_GPU_CI_ENABLED`),
      the same pattern as CUDA's `cuda-runtime`. See `gpu_backend_design.md`
      for the per-op hipBLAS/hipSOLVER mapping.
- [ ] Re-evaluate MAGMA, oneMKL SYCL, and Apple MPS against actual demand.

### Phase 6 — Release

- [ ] Tag 1.1 after Phase 3. Reserve 2.0 for removing the deprecated
      macros, the non-pivoting LU path, the deprecated no-workspace
      overloads, and [`include/memory/allocator.h`](../include/memory/allocator.h)
      — after which LinearAlgebra performs no allocation at all.
- [ ] Generate the README's backend × op coverage matrix from parity-suite
      results rather than maintaining it by hand.

## 7. Risks and open questions

- **Accelerate LAPACK availability**: the new interface requires macOS
  13.3+; older SDKs must keep the GEMM-only path. Decide the minimum
  supported macOS version before Phase 2.
- **Pivoting default flip (Phase 2)** changes numerical output of the
  default scalar build. Call it out in release notes; callers relying on
  the non-pivoting factor layout need a migration note.
- **`cusolverDnXgeev` support** (data types, eigenvector output format)
  must be checked against the targeted CUDA toolkits before Phase 4
  commits to it.
- **Eigen3 versus scalar precedence**: if both are available, Eigen3 should
  win for performance, but the scalar reference must remain compiled for
  parity testing (§4, item 2).
- **Workspace API surface**: adding a query plus an overload to every op
  that needs scratch roughly doubles the public declarations. Composite ops
  (pseudo-inverse via SVD, least squares via pseudo-inverse) must report
  the sum of their sub-ops' workspace and partition one buffer between
  them. Decide early whether alignment is fixed (for example 64 bytes) or
  reported per op.
- **GPU CI**: compile-only CUDA and HIP jobs catch build breaks but not
  numerical regressions; GPU correctness stays a manual, on-hardware check
  until a self-hosted CUDA (`LINALG_GPU_CI_ENABLED`) or ROCm
  (`LINALG_ROCM_GPU_CI_ENABLED`) GPU runner is available.

## 8. Recommended order and quick wins

Phases 0 and 1 come first: the LU pivoting default is the highest-impact
correctness issue, and the parity suite is what makes Phases 2–5 safe to
land. Cheapest high-value items:

- the OpenBLAS + LAPACKE CI job;
- the stale-comment fixes;
- pinning `MKL_INTERFACE`;
- checking whether the targeted CUDA toolkit provides `cusolverDnXgeev`.
