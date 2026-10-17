# Sparse matrix implementation and validation plan

Date: 2026-10-17. Status: future work only. This task adds Markdown
documentation, not numerical code, tests, dependencies, or build options.

Read the [integration rationale](sparse_matrix_integration.md) and
[technical design](sparse_matrix_design.md) before using this backlog.
Estimates are planning ranges, not delivery commitments.

## 1. Delivery scope and dependencies

The first useful release consists of canonical CSR views, COO assembly,
validation, native SpMV, identity/Jacobi preconditioning, and CG with a
true-residual report. It must work without MKL, CUDA, Accelerate, or a
mandatory Memory dependency.

Dependency order: workload inventory → storage contracts and build wiring
→ native products and assembly → CG → consumer pilot. SpMM and transpose
can follow the native foundation. Vendor adapters follow stable product
contracts and measured reuse. General solvers and sparse direct solvers
remain separate follow-ups.

## 2. Phased backlog

### Phase 0: confirm a useful workload and a healthy baseline

Estimated effort: 1–3 engineering days.

- Identify one consuming project and the exact dense allocation or banded
  limitation it needs to address.
- Collect representative shapes, stored counts, row-length distributions,
  scalar types, symmetry/SPD assumptions, RHS counts, and update patterns.
- Record existing runtime, peak memory, and numerical error. Include
  assembly, factorization, repeated solves, and transfers where applicable.
- Run the current supported build/test configurations available to the
  implementer. Record unrelated baseline failures before changing source.
- Compare the actual source against historical design documents; use
  current headers, build files, and CI as the starting point.
- Agree maximum problem size, residual/accuracy targets, memory budget,
  and whether faster runtime or enabling a larger problem is the objective.

Exit gate: one agreed pilot workload, reproducible baseline measurements,
and a written data/solver classification. A synthetic stencil can bootstrap
testing but cannot establish benefits for an unidentified consumer.

### Phase 1: stabilize storage and integration contracts

Estimated effort: 3–5 engineering days.

- Review borrowed CSR/COO views, lengths, empty shapes, canonical ordering,
  scalar/index support, alias restrictions, and error behavior.
- Define checked shape/byte arithmetic, alignment requirements, and the
  full-validation versus reusable validated-view distinction.
- Define exact workspace query behavior, output capacities, and returned
  actual counts for assembly.
- Register the proposed header/source trees, install/export behavior, and
  test/benchmark discovery in CMake and Bazel.
- Preserve existing dense build behavior when the feature is disabled.
  Decide feature defaults before publishing the new configuration.

Exit gate: native sparse-only and dense-only consumer smoke builds work;
installed headers and library symbols are available through each supported
packaging route; no accelerator SDK is required for the native path.

### Phase 2: native assembly and sparse products

Estimated effort: 4–7 engineering days.

- Implement full CSR validation and canonical COO-to-CSR assembly.
- Implement untransposed CSR SpMV with alpha/beta semantics for float and
  double and both supported compressed index widths.
- Add small explicit CSR-to-dense comparison support for tests; optional
  public conversion routines can follow without blocking the pilot.
- Verify zero-read behavior when beta = 0, zero-nnz output scaling, empty
  rows, and rejected overlapping buffers.
- Add meaningful product and assembly benchmarks, including setup costs.
- Add diagonal extraction needed by the next phase.

Exit gate: native results match independent small dense calculations and
algebraic properties within documented tolerances; malformed storage fails
before indexed access; native kernels allocate no hidden heap storage.

### Phase 3: first usable sparse solver

Estimated effort: 4–7 engineering days.

- Implement CG for declared SPD systems, with identity and positive Jacobi
  preconditioners, an initial guess, and explicit tolerances/iteration limit.
- Define the exact caller-owned vector workspace through one shared plan.
- Return termination reason, iteration count, and independently recomputed
  true residual, including zero-RHS and exact-initial-guess cases.
- Detect breakdown, nonfinite arithmetic, and observed nonpositive
  curvature. Test failure reports as carefully as successful solves.
- Validate against known solutions and existing dense solves on small
  well-conditioned SPD systems.

Exit gate: success is reported only when the true residual meets the
declared condition. Scaling and precision tests establish realistic
tolerances; the solver never silently switches to a dense method.

### Phase 4: pilot integration and reuse

Estimated effort: 3–6 engineering days, dependent on consumer access.

- Assemble one real operator directly into COO/CSR at its source.
- Keep the consumer's original dense or banded route selectable for
  comparisons and rollback.
- Separate pattern construction, value updates, and RHS updates.
- Benchmark one-shot and repeated solves with the same required accuracy.
- Record total memory, convergence, setup amortization, and end-to-end time.
- Add a regression fixture representative of the pilot without exposing
  private input data or committing large generated matrices.

Exit gate: the pilot satisfies its Phase 0 objective with reproducible
evidence. If it does not, investigate conditioning, preconditioning,
assembly, and row distribution before adding accelerators.

### Phase 5: SpMM, transpose, and broader CPU algorithms

Estimated effort: 3–6 engineering days for SpMM and transpose; general
solvers require their own estimates.

- Add row-major sparse-dense multiplication with explicit strides and
  multiple RHS columns.
- Add transpose products and/or explicit compressed transpose conversion.
- Measure repeated SpMV versus SpMM for the pilot's RHS counts.
- Choose a general-square or least-squares solver only when a consumer
  needs it; establish its stopping and breakdown semantics separately.

Exit gate: correct rectangular and padded-stride cases, transpose inner
product identity checks, and explicit capability documentation.

### Phase 6: optional vendor acceleration

Estimated effort: 3–6 engineering days for an initial oneMKL adapter and
5–10 for an initial cuSPARSE adapter with access to a suitable GPU.

- Add oneMKL Sparse handle creation, checked index conversion, optional
  analysis, reuse, error translation, and teardown.
- Add cuSPARSE discovery/linkage, device views, descriptor/workspace queries,
  explicit streams, plan invalidation, and safe lifetimes.
- Keep sparse-only CUDA builds independent of the dense cuBLAS toggle.
- Add capability reporting that distinguishes compiled support from a
  usable runtime device and a supported operation/type combination.
- Benchmark conversion/setup separately from warmed execution and total
  application time. Record vendor-managed metadata allocations.
- Consider Accelerate or SuiteSparse after a workload-specific evaluation,
  rather than treating every backend as required for the first release.

Exit gate: enabled adapters pass parity tests, native-only builds remain
independent, and CUDA has real runtime validation. Compile-only success
cannot establish numerical correctness or a speed benefit.

### Phase 7: release and later scope

Estimated effort: 2–4 engineering days for release preparation.

- Publish supported operations, formats, types, ownership, lifetimes,
  tolerances, backend limitations, and install instructions.
- Update the README's dense-only scope and backend table only when sparse
  behavior actually ships.
- Document pilot measurements and how to reproduce them.
- Keep SpGEMM, direct solvers, AAD, BSR, matrix-free operators, and advanced
  preconditioners as independently scoped milestones.

The Phase 0–4 native pilot is approximately 15–28 engineering days for one
engineer familiar with the project. Consumer availability, baseline build
problems, and difficult conditioning can extend it. Wider CPU operations,
vendor work, and release preparation are additional; do not promise all
backends as part of that pilot estimate.

## 3. Proposed repository change map

Every new path below is a planning target; it does not exist as a result of
this documentation task.

| Area | Proposed future location | Integration work |
|---|---|---|
| CPU sparse public contracts | `include/sparse_matrix_operation/` | Views, validation, assembly, products, solver options/results |
| Native CPU implementations | `src/sparse_matrix_operation/native/` | Portable baseline kernels |
| MKL adapter | `src/sparse_matrix_operation/mkl/` | Optional sparse handle and product wrappers |
| CUDA public contracts | `include/sparse_matrix_operation_gpu/` | Device views and stream-aware operations |
| CUDA implementation | `src/sparse_matrix_operation_gpu/` | Host-callable wrappers; `.cu` only when actual device kernels need it |
| Unit tests | `Testing/Cxx/TestSparse*.cxx` | Root-level test files match both current test globs |
| Benchmarks | `Testing/Benchmark/BenchmarkSparse*.cxx` | Existing benchmark target conventions |
| First-party build glue | `CMakeLists.txt`, `Cmake/`, `BUILD.bazel`, `bazel/` | Discovery, gating, definitions, linkage, exports |
| CI | `.github/workflows/ci.yml` | Native sparse, vendor parity, and packaging checks |

CMake currently discovers `.cpp` files recursively but only filters its
existing dense vendor directory names. New sparse vendor sources must be
excluded unless their dependency is enabled, or native-only builds could
accidentally compile vendor includes. Its `.cu` discovery is currently
controlled by the dense CUDA option and must be refactored for independent
sparse CUDA support.

Bazel lists specific header and source patterns, so new sparse trees need
explicit additions. Its CPU vendor toggles currently rely on toolchain
headers and libraries; do not claim hermetic sparse vendor support without
adding actual dependency resolution. Native CPU parity is required in both
systems. CUDA can remain explicitly CMake-only initially, matching the
existing GPU support boundary, with that limitation documented.

## 4. Proposed configuration

These names are reserved suggestions, not usable flags today.

| Proposed CMake flag | Proposed default | Purpose |
|---|---|---|
| `LINALG_ENABLE_SPARSE` | OFF during introduction | Enable native sparse kernels and contracts |
| `LINALG_ENABLE_MKL_SPARSE` | OFF | Sparse MKL capability; may share validated MKL discovery |
| `LINALG_ENABLE_CUSPARSE` | OFF | cuSPARSE independently of dense CUDA support |

Matching `LINALG_HAS_*` definitions should follow the existing 0/1
convention. Define whether vendor flags imply sparse support or require it;
prefer explicit configuration errors for inconsistent combinations over
silently ignored options. Share CUDA toolkit discovery where needed, but
enable the CUDA language only for sources that require it. The current
project may impose a newer CMake requirement for CUDA discovery than its
root minimum suggests; test and document the effective minimum rather
than assuming sparse wrappers work on every CMake version.

Share MKL discovery/interface settings with existing dense integration;
avoid loading incompatible LP64 and ILP64 variants into the same target.
Installed package metadata must rediscover enabled dependencies. A
native-only installation must be consumable without their SDKs.

Do not edit vendored code. Resolve adapter needs through first-party
wrappers and dependency configuration.

## 5. Verification matrix

| Category | Required cases | Evidence |
|---|---|---|
| CSR validity | Negative dimensions, invalid offsets, wrong final nnz, out-of-range/unsorted/duplicate columns, short capacities | Defined validation failures before reads/writes outside buffers |
| Empty structure | Zero dimensions, nnz = 0, empty first/middle/last rows, legal null zero-length arrays | Correct scaling/output and no invalid dereference |
| Assembly | Unsorted input, duplicates, cancellation, explicit zeros, optional pruning | Expected canonical coordinates and values |
| Products | Rectangular matrices, alpha/beta including zero, float/double, both index widths | Independent small dense reference |
| Layout | Padded B/C strides, multiple RHS, prohibited overlap | Correct logical results and untouched padding/sentinels |
| Transpose | Rectangular inputs, empty columns, repeated transpose | Bilinear identity uᵀAv = (Aᵀu)ᵀv and reference parity |
| Numerical limits | Ill-scaled values, NaN/Inf policy, subnormal/overflow behavior where specified | Defined results or failure classification |
| Integer safety | Boundary metadata and overflowing byte counts | Rejection without allocating impossible matrices |
| Workspace | Exact size, one byte short when nonzero, misalignment, null cases | Errors before corruption; allocation instrumentation for native paths |
| CG | Known SPD solutions, b = 0, exact initial guess, iteration limit, breakdown | Independently recomputed true residual and status |
| Reuse | New RHS, changed values, changed pattern, released/replaced buffers | Correct invalidation and refreshed analysis/preconditioner |
| CUDA | Nondefault streams, independent concurrent plans, device switching when available | Runtime parity and no unexpected product synchronization |
| Packaging | Native sparse on/off, installed consumer, enabled adapter exports | Clean configure/build/link from a separate consumer |

Use deterministic small fixtures, seeded synthetic families, and a small
set of representative real matrices. Existing dense kernels are useful
for small references; include hand-computed examples and algebraic
identities so a shared defect does not validate itself.

For a manufactured solve, select a known x, compute b = A × x, then check
both the residual and solution error on a well-conditioned fixture. A
small residual alone does not establish small solution error for an
ill-conditioned system. Do not use one universal tolerance for every size,
precision, and condition number.

Extend the established CMake Linux/macOS and Bazel native configurations.
Use existing sanitizer jobs for sparse buffer/lifetime checks, and the
coverage configurations including Windows where their toolchains permit.
Add MKL sparse cases to a dependency-equipped job. Keep CUDA compile-only
coverage, then add a GPU runner or a documented repeatable hardware test
before declaring runtime support. No platform is considered verified
solely because an adapter's dependency supports it.

## 6. Benchmark and decision protocol

Sweep problem size, density, row irregularity, scalar type, index width,
RHS count, and repeated-call count. Include diagonal/banded, stencil,
block-structured, irregular graph-like, and nearly dense cases. Compare
against the existing banded path where the input fits its structure.

Record compiler/flags, CPU/GPU, memory, OS, vendor versions, thread count,
matrix identity/seed, dimensions, nnz, tolerance, and iterations. Keep
threads and accuracy requirements comparable. Use repeated trials, warm
ups, and median plus spread; exclude initialization only from measurements
explicitly labeled warmed kernel time.

Report assembly, validation, index conversion, preprocessing,
preconditioning, execution, transfers, and peak memory separately, plus
their end-to-end total. Use device timing for queued GPU work and host
wall time for the whole workflow. Kernel throughput alone is insufficient.

Let S be sparse conversion/setup time, D be measured time per dense
operation, and P be time per comparable sparse operation. When D > P,
sparse becomes cheaper after a repeated-call count R satisfying
R > S / (D − P), assuming the dense alternative has no comparable setup.
If both paths have setup, use their setup-cost difference. If D ≤ P,
reuse alone does not create a runtime advantage, though memory may still
justify sparse storage. Iterative solver comparisons must use complete
solve time at the required accuracy rather than treating one SpMV as a
whole solve.

Proposed pilot acceptance: meet the consumer's accuracy target and either
enable the required size within its memory budget or demonstrate a useful
runtime improvement agreed in Phase 0. For example, a 20% runtime
improvement may be a reasonable target only when it exceeds measurement
noise and integration cost. It is a target, not a prediction.

## 7. Risks, rollout, and rollback

| Risk | Mitigation / decision |
|---|---|
| Sparse storage is smaller but slower | Measure total cost and retain dense/banded selection |
| CG used on an unsuitable matrix | Require SPD classification; report breakdown; choose a later suitable solver |
| Weak preconditioning dominates runtime | Measure iteration counts before optimizing products or moving to GPU |
| Converted indices erase memory savings | Include conversion buffers in peak memory; choose supported width deliberately |
| Reused metadata becomes stale | Separate pattern and value revisions; test every invalidation case |
| Sparse factorization fills memory | Defer direct solvers; measure fill and honor a consumer memory budget |
| Vendor setup allocates internally | Disclose metadata ownership and lifetimes; retain strict native path |
| GPU transfers dominate | Preserve device residency and report end-to-end timing |
| Existing glob compiles disabled adapters | Add explicit build gating and sparse-off CI |
| Historical docs overstate current behavior | Keep proposed APIs labeled and use source-backed capability tables |

Introduce sparse support behind explicit configuration and explicit
consumer selection. Pilot with a single operator, then expand based on
measurements. Keep pattern metadata close to the consumer that knows when
it changes; keep domain-specific assembly rules outside the kernel library.

Rollback means choosing the previous consumer path or disabling sparse
support and rebuilding. Do not silently fall back to an enormous dense
allocation when a sparse solve fails. Report the failure and let the
consumer decide how to proceed.

## 8. Completion criteria

The future native release is complete when its documented formats and
operations exist, both supported CPU build systems and installed-consumer
checks pass, numerical and memory contracts are verified, one real pilot
meets its agreed goal, and user documentation reflects shipped behavior.
Vendor support is complete separately for each adapter after runtime
verification on the stated platform.

For the current request, completion means delivering these linked design
and planning documents and validating their formatting and links. No
implementation approval, backend installation, or source change is implied.
