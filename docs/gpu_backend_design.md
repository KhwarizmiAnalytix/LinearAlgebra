# LinearAlgebra backend design: full MKL / BLAS / cuBLAS / cuSOLVER coverage

Status: §8 phases 1–3 implemented and verified against real CUDA hardware
(handle pool + stream parameter, the `lu_decomposition_gpu` layout fix,
`matrix_inversion_gpu`, `svd_decomposition_gpu`). Phase 4 (multi-RHS and
batched extensions) is not yet started.

**Standing principle: `LinearAlgebra` does not manage memory.** It is a
kernel library — factorizations, solves, products, transposes — not an
allocator. Every entry point, CPU and GPU alike, takes and returns bare
`T*` the caller allocates and owns; the caller decides how (`cudaMalloc`,
`malloc`/`new`, a pool, a third-party allocator — this library neither
knows nor cares). See §2 for the full statement and what it rules out.

## 1. Goal

Bring the four optional backends declared in `CMakeLists.txt` —
`LINALG_ENABLE_MKL`, `LINALG_ENABLE_BLAS`, `LINALG_ENABLE_CUBLAS` (which
covers both cuBLAS and cuSOLVER) — to full, matched op coverage across CPU
and GPU, and make chained GPU calls run start-to-finish on device memory
with no implicit host round-trip. Concretely:

1. Every op that exists for the CPU (`cholesky_decomposition`,
   `lu_decomposition`, `linear_solver`, `matrix_invert`/`matrix_determinant`,
   `matrix_multiplication`, `matrix_transpose`, `svd_decomposition`) gets a
   `linalg::gpu::*` counterpart. Today `matrix_invert`/`matrix_determinant`
   and `svd_decomposition` have none.
2. GPU calls stop being pinned to a single implicit stream, and stop
   producing a factorization (`lu_decomposition`) whose on-device layout is
   incompatible with the CPU convention. The existing per-call
   `cudaMalloc`/`cudaFree` workspace pattern stays exactly as it is today
   (see §2) — this plan does not change how workspace is allocated, only
   that its return value is finally checked.
3. A caller can run a multi-step pipeline (e.g. factorize → solve →
   multiply) with every intermediate value staying on the device, ending in
   exactly one deliberate host transfer (or none, if the result is consumed
   by another GPU op or another library sharing the same CUDA context). The
   caller supplies and owns every device pointer, as today; nothing here
   introduces a library-side ownership type.

## 2. Non-goals

- **No memory management design.** Every CPU and GPU entry point continues
  to take and return bare `T*` the caller allocates and frees itself
  (`cudaMalloc`/`cudaFree`, `malloc`/`new`, or any allocator the caller
  chooses) — exactly the contract every `linalg::*`/`linalg::gpu::*`
  function already has today. This plan does not introduce a caching
  allocator, an owning device/host buffer type, or a dependency on any
  external memory library (XSigma's `Library/Memory` or otherwise). Where a
  GPU op needs internal scratch (factorization workspace, a pivot array),
  it keeps allocating that scratch itself with plain `cudaMalloc`/`cudaFree`
  around the call, the same way `cholesky_decomposition_gpu.cxx` and
  `linear_solver_gpu.cxx` already do — the only change there is adding the
  error check that call already lacks (§4.3).
- No autograd, no dtype-erasure/boxed dispatch, no `Tensor` type. This stays
  a plain, explicit, `T*`-and-dimensions C++ library.
- No reintroduction of a runtime backend registry for the CPU path. Commit
  `b09f66e` deliberately replaced a PyTorch-style `dispatch_stub`
  function-pointer registry with a single `#if LINALG_ENABLE_MKL / #elif
  LINALG_ENABLE_BLAS / #else` chain compiled directly into each op's `.cxx`,
  to remove per-call atomic loads and the whole-archive linker workaround
  the registrars needed. That decision stands; nothing here touches CPU
  dispatch.
- No Bazel GPU target. `BUILD.bazel` already documents that
  `include/matrix_operation_gpu/` is CMake-only ("CMake is this repo's
  primary, exercised build system for ... the cuBLAS/cuSOLVER GPU
  backend"); this plan does not change that split. CMake already wires
  `CUDA::cudart`, `CUDA::cublas`, and `CUDA::cusolver` and globs
  `include/*.cxx`/`include/*.cu` when `LINALG_ENABLE_CUBLAS` is `ON`, so new
  GPU `.cxx`/`.cu` files need no `CMakeLists.txt` change to be picked up.
- No MAGMA, no multi-node, no half/complex types. Stay with `float`/`double`
  and single-GPU, matching the existing scope.

## 3. What "rely on the PyTorch project" means here

PyTorch's ATen linear-algebra layer (`aten/src/ATen/native/cuda/linalg/`,
`aten/src/ATen/cuda/CUDABlas.cpp`, `CUDASolver.cpp`) is the reference for
the *mechanics* this repo is currently missing, not for its dispatch
architecture (which this repo already tried and rejected — see §2) and not
for its allocator (out of scope by §2 — this plan stays on raw pointers).
Two PyTorch patterns are worth importing, each scoped down to fit this
codebase and this plan's raw-pointer contract:

| PyTorch mechanism | What it solves | Scoped-down equivalent proposed here |
|---|---|---|
| `getCurrentCUDABlasHandle()` / `getCurrentCUDASolverDnHandle()`, keyed by `(device, stream)` | A single `thread_local` handle (today's `cuda_handle.h`) is silently wrong across `cudaSetDevice` calls and never rides the caller's stream | §5.1 — handle pool keyed by device ordinal + stream |
| `cloneBatchedColumnMajor` | cuSOLVER is column-major; PyTorch materializes a device-side transpose rather than accepting a semantically different factorization | §5.2 — use the library's own `matrix_transpose_gpu` kernel (already device-only, already exists) to fix `lu_decomposition_gpu`'s A-vs-A^T mismatch, instead of leaving it as a documented caveat |
| Batched linear algebra over a leading batch dimension (`torch.linalg.solve` on a `(*, n, n)` tensor lowers to `getrfBatched`/`gesvdaStridedBatched`) | Many independent small systems is the common case in practice, not one big matrix | §5.4 — `getrfBatched`/`potrfBatched`/`gesvdjBatched` overloads alongside the existing single-matrix ones, and replacing the hand-written batched GEMM kernel with `cublasXgemmStridedBatched` |

Everything else about ATen — `TensorIterator`, the dtype-macro dispatcher,
`DispatchStub`, `CUDACachingAllocator`, autograd formulas — is out of scope
by §2.

## 4. Current state (as of this design)

### 4.1 Op coverage matrix

| Op | CPU (scalar/BLAS/MKL) | GPU (cuBLAS/cuSOLVER) |
|---|---|---|
| `matrix_multiplication` | yes | yes (single + a hand-written batched kernel) |
| `matrix_transpose` | yes | yes |
| `cholesky_decomposition` | yes | yes |
| `lu_decomposition` | yes | yes, but returns a **different factorization** than the CPU op (see §4.2) |
| `linear_solver` | yes, single RHS vector | yes, single RHS vector only |
| `matrix_invert` / `matrix_determinant` | yes | **missing** |
| `svd_decomposition` | yes | **missing** |

### 4.2 Known correctness caveat: `lu_decomposition_gpu`

`include/matrix_operation_gpu/lu_decomposition_gpu.h` documents that
`cusolverDnXgetrf` factors the column-major reading of the row-major buffer
— i.e. `A^T`, not `A` — so it returns `P · A^T = L' · U'`, not the CPU's
row-major `P · A = L · U`. `linear_solver_gpu.cxx` works around this
internally for its own `lu_solve()` by passing `CUBLAS_OP_T` to
`cusolverDnXgetrs`, but the standalone `linalg::gpu::lu_decomposition` entry
point has no such correction, and `TestGpuMatrixOperations.cxx`'s
`LuDecompositionSmoke` test explicitly only checks `info == 0`, not the
factorization's numerical content, because "that factorization is not meant
to be compared against the CPU convention." §5.2 fixes this.

### 4.3 Other gaps

- Every GPU factorization (`cholesky_decomposition_gpu.cxx`,
  `linear_solver_gpu.cxx`) calls `cudaMalloc`/`cudaFree` for its workspace
  (and, for LU, its pivot array) on every single invocation, and does not
  check `cudaMalloc`'s return value. Per §2, the per-call allocation pattern
  itself is unchanged by this plan — only the missing `LINALG_THROW` on a
  failed `cudaMalloc` is added, at each existing call site, the same way
  every other failure in these files is already reported.
- `detail::cublas_handle()` / `detail::cusolver_handle()`
  (`include/common/cuda_handle.h`) are one `thread_local` instance each,
  with no device or stream key. A thread that calls `cudaSetDevice` between
  two library calls keeps using the first device's handle. No call site
  ever calls `cublasSetStream`/`cusolverDnSetStream`, so every op runs on
  the legacy default stream regardless of what stream the caller is
  otherwise using.
- `linear_solver` (both CPU and GPU) only solves a single right-hand-side
  vector (`x`, no `nrhs`/`ldx`). There is no batched entry point anywhere in
  `matrix_operation_gpu/` except the hand-written
  `batched_matrix_multiplication` kernel, which does not use cuBLAS's own
  batched GEMM.

## 5. Proposed additions

### 5.1 Per-device, stream-aware handle pool

Replace the two `thread_local` handles in `include/common/cuda_handle.h`
with a pool keyed by the currently active CUDA device (`cudaGetDevice`):

```cpp
namespace linalg { namespace detail {

// One {cublasHandle_t, cusolverDnHandle_t} pair per device ordinal seen so
// far by this thread, created lazily on first use per device (same
// lazy-creation cost tradeoff as today, just no longer wrong across
// cudaSetDevice). Still thread_local: cuBLAS/cuSOLVER handles are not
// safe to share across concurrently-executing threads (unchanged from the
// header's existing comment). Device selection around handle lookup/
// creation is a plain save-`cudaGetDevice`/`cudaSetDevice`-restore RAII
// guard local to this file — ordinary execution-context bookkeeping, not
// memory management, so it needs no new dependency (§2).
cublasHandle_t     cublas_handle_for_current_device();
cusolverDnHandle_t cusolver_handle_for_current_device();

} }
```

Add stream binding as an explicit, opt-in overload rather than a hidden
global — every `linalg::gpu::*` function gains a
`cudaStream_t stream = nullptr` trailing default parameter (`nullptr` =
legacy default stream, preserving today's behavior exactly for existing
callers), and each op's `.cxx` calls `cublasSetStream`/
`cusolverDnSetStream` on the pooled handle before issuing work. This keeps
the change additive (no signature breaks for `stream = nullptr` callers)
while letting a caller pipeline several `linalg::gpu::*` calls on their own
stream, or interleave them with other libraries sharing the same CUDA
context (e.g. issuing work alongside a caller-owned CUDA graph capture).

### 5.2 Fix `lu_decomposition_gpu`'s layout mismatch

Use the library's own `linalg::gpu::matrix_transpose` (already a pure
device-to-device kernel, no host traffic) to make `lu_decomposition_gpu`
return the same `P · A = L · U` convention as the CPU op, instead of
leaving `A^T`'s factorization as a documented-but-uncorrected caveat:

1. Transpose the input in place on the device (`matrix_transpose(n, n,
   m)`).
2. Run `cusolverDnXgetrf` as today — this now factors the *actual* `A`
   (since the buffer handed to cuSOLVER is `A`'s transpose, and cuSOLVER
   reads it column-major, cuSOLVER's column-major view of `transpose(A)`
   *is* `A` read row-major... equivalently: transpose once up front so the
   existing "factor A^T" behavior now factors the caller's real `A`).
3. Transpose the resulting packed `L\U` buffer back, so the row-major
   layout callers get matches `lapacke_?getrf`'s row-major packed output
   bit-for-bit in convention (though not implementation).

This costs two device-side transpose kernel launches per call — cheap
relative to `O(n^3)` factorization work, and it never touches host memory,
so it does not compromise §1's "stays on device" requirement. Update
`linear_solver_gpu.cxx`'s `lu_solve()` to match (it can drop its
`CUBLAS_OP_T` workaround once the factor going in is already `A`, not
`A^T`), and change `TestGpuMatrixOperations.cxx`'s `LuDecompositionSmoke`
into a real correctness test (compare `P·A` against `L·U` reconstructed
from `linalg::gpu::lu_decomposition`'s output, the same way
`TestLUDecomposition.cxx` already checks the CPU path).

### 5.3 New GPU ops

**`matrix_inversion_gpu.h`/`.cxx`** (`linalg::gpu::matrix_invert`,
`linalg::gpu::matrix_determinant`). cuSOLVER has no direct `getri`
equivalent, so mirror the CPU op's actual strategy
(`matrix_inversion.cxx` already computes the inverse via
factor-then-solve-against-identity, not a dedicated inversion routine):
`cusolverDnXgetrf` (via the corrected §5.2 path) followed by
`cusolverDnXgetrs` with the identity matrix as the batched right-hand
side (`nrhs = n`, needs the §5.4 multi-RHS `getrs` support). The identity
matrix is built directly on the device (a small fill kernel, or
`cudaMemsetAsync` plus a strided unit-diagonal write) into a buffer the
caller supplies (or the op allocates and frees itself with plain
`cudaMalloc`/`cudaFree`, per §2) — never staged through the host.
`matrix_determinant` reads the product of the factor's diagonal together
with the pivot permutation's sign, exactly as the CPU path does, evaluated
with a small device-side reduction kernel (or a `thrust`/`cub` reduction if
either becomes a dependency — otherwise a manual one-block kernel, since
`n` is a single matrix dimension, not a large array) so the scalar result
requires only one device→host copy of a single value, not the whole
factor.

**`svd_decomposition_gpu.h`/`.cxx`** (`linalg::gpu::svd_decomposition`).
Two cuSOLVER paths, both column-major like `getrf`, so both need the same
transpose-in/transpose-out treatment as §5.2 applied to `A`, `U`, and
`V^T`:

- `cusolverDnXgesvd` — general dense SVD, matches the CPU op's semantics
  and value range most directly (LAPACK-equivalent accuracy). Use this as
  the default, matching the CPU backend's own accuracy characteristics so
  results are comparable across backends within test tolerance.
- `cusolverDnXgesvdj` (one-sided Jacobi) as a documented, opt-in faster
  path for small/medium matrices where Jacobi's accuracy is adequate —
  expose it as a separate overload or an `svd_algorithm` enum parameter
  rather than silently swapping algorithms under the existing signature,
  since it changes numerical behavior at the margins.

Both land in `include/matrix_operation_gpu/svd_decomposition_gpu.h`,
mirroring the CPU header's row/column/`lda`/`ldu`/`ldv` signature shape.

### 5.4 Multi-RHS and batched extensions

- **`linear_solver_gpu`**: extend `x`'s contract from "one column" to
  `nrhs` columns with an explicit `ldx`/`nrhs` parameter pair (default
  `nrhs = 1` preserves every existing call site unchanged), passed straight
  through to `cusolverDnXgetrs`/`cusolverDnXpotrs`'s own `nrhs` parameter,
  which already supports this — today's code hardcodes `nrhs = 1`
  needlessly.
- **Batched factorizations** for the "many small independent systems" case
  (the actual common case in practice, per §3's table): add
  `linalg::gpu::batched_cholesky_decomposition`,
  `batched_lu_decomposition`, `batched_linear_solver` built on
  `cusolverDnXpotrfBatched`/`cusolverDnXgetrfBatched` +
  `cusolverDnXgetrsBatched`-equivalent host-side loop (cuSOLVER's batched
  `getrs` is not a single-call batched primitive the way `getrf`/`potrf`
  are as of common cuSOLVER versions — confirm against the target CUDA
  toolkit version during implementation and fall back to a stream-per-item
  loop over the pooled handle from §5.1 if no batched `getrs` exists in the
  targeted toolkit).
- **`batched_matrix_multiplication`**: replace the hand-written one-thread-
  per-output-element kernel in `matrix_multiplication_batched_gpu.cu` with
  `cublasXgemmStridedBatched` (uniform stride across same-size matrices,
  exactly this op's existing contract), which uses cuBLAS's tuned kernels
  (tensor cores where applicable) instead of a naive kernel. Keep the
  existing entry point signature; only the implementation changes. Validate
  the switch does not regress the very-small-`dim` case (`dim = 9` in the
  existing test) against the current kernel's measured throughput before
  removing it — cuBLAS's batched GEMM has per-call setup overhead that a
  trivial fixed-size kernel can beat for very small, very numerous
  matrices; if that holds, keep both and pick by a size heuristic
  (documented, not hidden) rather than deleting the working kernel outright.

## 6. Public API surface summary

New headers under `include/matrix_operation_gpu/` (all guarded by
`#ifdef LINALG_ENABLE_CUBLAS`, all device-pointer-only, all following the
existing file's documentation style — explicit "no host memory read or
written" / "does not synchronize" notes on every declaration, and, per §2,
every parameter a bare `T*`/`int*` the caller owns, exactly like every
existing `matrix_operation_gpu/*.h` entry point):

- `matrix_inversion_gpu.h`
- `svd_decomposition_gpu.h`
- `batched_cholesky_decomposition_gpu.h`, `batched_lu_decomposition_gpu.h`,
  `batched_linear_solver_gpu.h`

Modified, signature-compatible (existing call sites keep compiling
unchanged via trailing defaults):

- `linear_solver_gpu.h` — `nrhs`/`ldx`, trailing `cudaStream_t stream =
  nullptr`.
- Every other `matrix_operation_gpu/*.h` — trailing `cudaStream_t stream =
  nullptr`.
- `cuda_handle.h` — device-keyed pool instead of one `thread_local`
  instance (internal `detail::` API, not publicly visible, so this is not
  an ABI/API break for callers).

Unchanged: every CPU header in `include/matrix_operation/`, the compile-
time `#if LINALG_ENABLE_MKL / #elif LINALG_ENABLE_BLAS / #else` dispatch
pattern in each op's `.cxx`, `include/memory/allocator.h` (host scratch
allocator — untouched, out of scope per §2), `include/common/context.h`'s
read-only introspection surface (`has_mkl()`/`has_blas()`/`has_cuda()` stay
as they are; no `has_batched_solve()`-style capability flags are needed
since every addition here is compiled in whenever `LINALG_ENABLE_CUBLAS`
is, exactly like the existing GPU ops).

## 7. Testing plan

Per `.augment/rules/testing.md` conventions (Google Test, `Test*.cxx` under
`Testing/Cxx/`, glob-registered in both CMake and Bazel where applicable —
Bazel does not build the GPU tree, per §2, so these are CMake/CTest-only,
same as `TestGpuMatrixOperations.cxx` today):

1. **New op correctness**: `matrix_invert`/`matrix_determinant` and
   `svd_decomposition` GPU tests compare against the same CPU reference
   values `TestMatrixInversion.cxx`/`TestSVDDecomposition.cxx` already
   compute, at `GpuTolerance<T>` (existing `5e-3f`/`1e-9` thresholds).
   Device buffers in the test file continue to use the existing test-only
   `device_buffer<T>` helper already in `TestGpuMatrixOperations.cxx` — no
   library-side buffer type is added (§2).
2. **LU layout fix regression**: `LuDecompositionSmoke` becomes a real
   `P · A = L · U` check (§5.2), reusing the existing SPD/diagonally-
   dominant matrix builders already in the test file.
3. **Multi-RHS**: extend `linear_solver_gpu_test` to `nrhs > 1` (a random
   `n x k` right-hand side, not just a vector), checked against the same
   `matmul`/`max_abs_diff` helpers already in the file.
4. **Batched ops**: one test per batched entry point, `count` independent
   small SPD/general matrices built the same way `spd_matrix()` already
   does, each checked independently against its own CPU reference — mirrors
   the existing `BatchedMatrixMultiplication` test's structure.
5. **Stream parameter**: a test that runs two independent solves on two
   distinct caller-created streams concurrently and confirms both complete
   correctly (does not attempt to assert actual concurrency/overlap, which
   is timing-sensitive and unsuited to a correctness test — the goal is
   "does not deadlock or corrupt," not "is measurably faster").
6. **Existing coverage stays green**: `TestGpuMatrixOperations.cxx`'s
   current tests (`MatrixMultiplication`, `MatrixTranspose`,
   `BatchedMatrixMultiplication`, `CholeskyDecomposition`,
   `LinearSolverCholesky`, `LinearSolverLU`) must keep passing unmodified
   except where §5.2/§5.4 explicitly change their semantics.

GPU tests remain opt-in: `.github/workflows/ci.yml` does not currently
build with `LINALG_ENABLE_CUBLAS` (no GPU hardware in CI runners), so this
plan does not claim CI coverage for any of the above — they are
local/manual verification on real CUDA hardware, exactly like today's GPU
test file already is (`LINALG_ENABLE_TESTING`/`LINALG_ENABLE_CUBLAS` both
`ON`, run via `Scripts/setup.py`, on a machine with a CUDA device).

## 8. Phased rollout

Ordered by dependency, each phase independently buildable/testable:

1. **DONE** — **§5.1 handle pool + stream parameter** — pure infrastructure,
   no public API change beyond the additive trailing `stream` default,
   fixes the multi-GPU-handle correctness gap immediately. Bundled in the
   §4.3 unchecked-`cudaMalloc` fix at the same time (a `LINALG_THROW` added
   at each existing call site) since both touched the same three files
   (`cholesky_decomposition_gpu.cxx`, `lu_decomposition_gpu.cxx`,
   `linear_solver_gpu.cxx`). Verified: `MathGpu.StreamParameter` (two
   independent solves on two caller-created streams) plus every existing
   GPU test, on real CUDA hardware.
2. **DONE** — **§5.2 LU layout fix** — corrects existing behavior before
   anything new builds on top of the (previously wrong) factorization
   convention. `include/common/cuda_fwd.h` (a forward declaration of
   `cudaStream_t` so public headers can take a stream parameter without
   requiring the CUDA toolkit on a CPU-only build's include path) was added
   as part of this work. Verified: `MathGpu.LuDecomposition` now checks
   real `P · A = L · U` reconstruction (previously only a smoke test),
   passing on real hardware.
3. **DONE** — **§5.3 new ops** (`matrix_inversion_gpu`, `svd_decomposition_
   gpu`) — closes the CPU/GPU op-coverage gap from §4.1's table.
   `matrix_invert`/`matrix_determinant` collapse the CPU's `*_UPFRONT_*`
   variants into their non-upfront counterparts, matching
   `linear_solver_gpu`'s existing precedent (cuSOLVER always re-factors;
   there is no "assume already factored" GPU path). `svd_decomposition_gpu`
   additionally discovered and worked around a real constraint this design
   doc did not anticipate: cuSOLVER's classic `gesvd` only supports
   `m >= n`, unlike LAPACK's `gesvd` — the `rows < columns` case is handled
   by factoring `A^T` internally and relabeling `U`/`V` (see
   `svd_decomposition_gpu.cxx`). Also ships a documented RESTRICTION not in
   the original plan: `lda`/`ldu`/`ldv` must be tightly packed
   (`lda == columns`, `ldu == min(rows, columns)`, `ldv == columns`) — a
   general-stride implementation was judged not worth the added complexity
   for this pass. Verified: `MathGpu.MatrixInversion` (LU- and
   Cholesky-based inversion agree, `A · A⁻¹ ≈ I`, both backends' determinants
   agree) and `MathGpu.SVDDecomposition` (`A ≈ U · diag(S) · Vᵀ` for both a
   tall and a wide matrix), on real hardware.
4. **Not started** — **§5.4 multi-RHS and batched extensions** — highest
   implementation effort (cuSOLVER batched-API surface varies more across
   toolkit versions than the single-matrix APIs used elsewhere in this
   plan), left for last so its toolkit-version research does not block the
   rest.

Each phase ships with its own `Testing/Cxx` coverage from §7 before the
next phase starts, per this repository's verification conventions
(`.claude/skills/session-checklist/SKILL.md`).

## 9. Open questions / risks

- **cuSOLVER batched `getrs`**: as noted in §5.4, whether a true batched
  `getrs` exists (vs. `getrf`/`potrf`, which are batched in all reasonably
  recent cuSOLVER releases) needs confirming against whatever CUDA toolkit
  version this repository targets in CI/dev images before committing to the
  single-call-batched design; the per-item-loop fallback changes the
  performance story but not the public signature.
- **Per-call workspace allocation stays a known cost, by design (§2)**: a
  caller that runs the same factorization in a tight loop still pays a
  `cudaMalloc`/`cudaFree` pair every call, exactly as today — this plan
  does not add caching. A caller with that access pattern is expected to
  manage the cost itself (e.g. by batching via §5.4 instead of looping),
  not rely on the library to amortize it.
- **Multi-GPU is unexercised**: §5.1's device-keyed handle pool makes
  multi-GPU *correct* but nothing in `Testing/Cxx` currently runs on more
  than one device — a multi-GPU CI runner is out of scope for this plan;
  the design should not regress single-GPU behavior while it goes
  untested on real multi-GPU hardware.
- **cuBLAS batched GEMM vs. the existing hand-written kernel** (§5.4): the
  size heuristic threshold (if kept dual-path) needs an actual benchmark on
  target hardware, not an assumed cutoff — do not hardcode a guessed `dim`
  threshold without measurement.
- **Jacobi SVD (`gesvdj`) accuracy**: needs validation against this
  project's existing SVD test tolerances before being offered as anything
  other than an explicitly-opt-in alternative algorithm (§5.3).
