# Plan to fuse CPU and GPU matrix operations

Status: proposal only. This document introduces no source, API, or build changes.

## Objective

Bring `matrix_operation` and `matrix_operation_gpu` into one operation-oriented
API and source hierarchy, with explicit backend selection and shared mathematical
contracts. Preserve CPU-only builds, caller-owned memory, CUDA stream behavior,
and existing entry points throughout migration.

Fusion means sharing operation names, contracts, validation infrastructure, and
backend organization. CPU algorithms, vendor adapters, and CUDA kernels remain
separate implementations. This work does not imply automatic transfers, runtime
backend selection, or CUDA support for every CPU operation.

## Current structure and constraints

The inspected source has two distinct public interfaces:

- CPU operations use `linalg::<operation><T, linalg_backend>::apply` and
  `workspace_size`. `include/common/backend.h` lists `native`, `blas_lapacke`,
  and `mkl`; callers explicitly select the backend.
- GPU operations use overloaded `linalg::gpu::*` functions for float and double,
  with separate workspace queries, CUDA device pointers, and a trailing stream.
  Some operations expose a device `int* info` rather than a CPU return status.
- Both implementations already use caller-owned scratch and the shared
  `include/common/workspace_arena.h`. `cuda_workspace.h` is a compatibility
  include, not a separate allocator.
- CPU and GPU contracts already share enums such as `norm_type` and
  `linear_solver_type`, but some GPU headers obtain them by including CPU
  operation headers.
- GPU reductions such as norm and trace return host scalars and synchronize.
  GPU multiplication and many factorizations queue work on the supplied stream.
  A unified API must retain this distinction.
- CUDA SVD requires packed row-major buffers; CPU vendor implementations and
  other operations may support wider strides. Existing header commentary is not
  uniformly current: audit implementation and tests before defining guarantees.
- Public dimension aliases mix signed `linalg_int` and unsigned `linalg_long`.
  `linalg::index_t` already provides the intended signed replacement. CUDA pivot
  and status arrays use `int`, which must not be reinterpreted as `index_t`.

Relevant references: [CPU backend definitions](../include/common/backend.h),
[CPU multiplication](../include/matrix_operation/multiplication.h),
[GPU multiplication](../include/matrix_operation_gpu/multiplication_gpu.h),
[GPU SVD](../include/matrix_operation_gpu/svd_decomposition_gpu.h),
[backend roadmap](backend_roadmap.md), and [GPU design](gpu_backend_design.md).
Treat older design prose as context; current declarations, implementations, and
tests determine the migration baseline.

## Recommended destination

### One public operation family

Extend the existing explicit backend model with a CUDA backend identifier,
provisionally `linalg_backend::cuda`. Keep existing CPU enum values stable.
Use the existing operation templates as the canonical family and add CUDA
specializations. Their members may accept backend-specific execution arguments;
do not force GPU execution into a synchronous CPU signature.

For example, multiplication remains one named operation with CPU and CUDA
specializations. CPU calls retain host buffers and the current workspace
contract. CUDA calls require device buffers and a CUDA execution descriptor.
The descriptor carries the stream and, where required, caller-owned device
status storage. It owns no memory and performs no synchronization on destruction.
Keep its declaration usable without CUDA toolkit headers, following `cuda_fwd.h`.

Specify each specialization's signature explicitly. A shared template name does
not promise that one unchanged call expression works on every backend. Any
later generic algorithm must constrain the execution and memory contracts it uses.

### Shared contracts with explicit memory domains

Define lightweight, non-owning host/device matrix and workspace descriptors for
the new CUDA-facing interface. Include dimensions, row stride, pointer, and byte
capacity as appropriate. Preserve current CPU signatures during this migration;
typed CPU overloads can be a later additive change.

Device descriptors identify the intended address space; wrapping a pointer does
not prove its allocation type. Document caller responsibility and avoid implicit
pointer inspection, allocations, staging buffers, or transfers in hot paths.

Extract genuinely shared enums and options into neutral common headers, leaving
compatibility includes in the old headers. Share dimension, stride, overflow, and
workspace-capacity checks where contracts match. Never dereference device
buffers in common host validation.

### Backend source organization

Use the following proposed layout, keeping the current public CPU header paths:

| Location | Responsibility |
| --- | --- |
| `include/matrix_operation/<operation>.h` | Canonical operation declarations and shared contract |
| `include/matrix_operation/cuda/<operation>.h` | CUDA specialization declarations when a separate header improves isolation |
| `include/matrix_operation/detail/` | Declaration helpers without vendor headers |
| `src/matrix_operation/native/` | Native CPU kernels |
| `src/matrix_operation/blas/` | CBLAS/LAPACKE adapters |
| `src/matrix_operation/mkl/` | MKL adapters |
| `src/matrix_operation/cuda/` | Moved CUDA `.cpp` adapters and `.cu` kernels |
| `src/matrix_operation/` | Existing composite operations and shared orchestration |
| `include/matrix_operation_gpu/` | Temporary forwarding headers for old includes |

Keep vendor headers private. Reuse mathematical orchestration only when memory,
status, and stream semantics agree; shared names alone do not justify merging
CPU and GPU implementation bodies.

## Operation inventory and migration order

| Group | Existing coverage | Planned treatment |
| --- | --- | --- |
| Multiplication, transpose | CPU and GPU | Pilot unified declarations, selection, and compatibility adapters |
| LU, Cholesky, solve, inversion | CPU and GPU | Preserve factor layout, pivot widths/conventions, mutation, and device status |
| QR, SVD, symmetric eigen | CPU and GPU | Document shapes, packing, ordering, convergence, and workspace restrictions |
| Pseudo-inverse, least squares | CPU and GPU | Migrate after SVD and multiplication dependencies |
| Norm, rank, trace | CPU and GPU | Preserve host-result synchronization and backend-specific scratch requirements |
| Determinant | CPU header; GPU entry points in inversion header | Inventory actual symbols and give determinant a coherent canonical declaration |
| Batched multiplication | GPU header only | Retain CUDA-only support; adding CPU batching is separate work |
| General eigen, condition number, power, exponential, banded operations | CPU headers without matching GPU headers | Retain current support and explicitly mark unsupported CUDA combinations |

This inventory describes header-level coverage, not complete availability for
every CPU backend. Phase 1 must record backend and scalar support per symbol.
Do not infer missing implementations from names or invent host fallback paths.

## Execution and numerical rules

1. Backend selection stays explicit. Unsupported combinations must fail clearly
   through constrained declarations, deleted members, or a documented compile-time
   diagnostic rather than silently selecting another backend.
2. Preserve row-major interpretation, transpose behavior, leading dimensions,
   input mutation, output shapes, and supported aliasing per operation.
3. Preserve CPU status/exception behavior. CUDA launch/vendor API errors remain
   distinct from numerical failure reported asynchronously through device info.
4. Preserve existing synchronization points. CUDA device status, matrix outputs,
   and scratch remain alive until the submitted stream work completes. Host
   scalar results retain their documented blocking behavior.
5. Queries remain backend-specific and return bytes. Include alignment, overflow,
   nested-operation scratch, and any vendor query requirements. Do not assume
   CPU and CUDA workspaces have interchangeable sizes or address spaces.
6. Preserve LU pivot conventions, including native pivoting build options.
   Check decompositions by reconstruction and residuals; do not demand identical
   factors, singular-vector signs, or eigenvectors across implementations.
7. Use `index_t` for new descriptors and checked narrowing at vendor boundaries.
   Keep legacy signatures and vendor pivot types intact until a separately
   versioned API change permits removal.

## Phased implementation

### Phase 1: establish the contract baseline

- Inventory every CPU template specialization and GPU exported overload, including
  workspace queries and determinant functions exposed from inversion headers.
- Record scalar types, backend availability, dimensions, strides, mutation,
  status, synchronization, workspace, and allocation ownership in a review table.
- Resolve contradictions between comments and implementation, including SVD
  strides/constness, solver scratch ownership, and stale dispatch descriptions.
- Capture existing public symbols and representative installed-consumer calls.

Exit: a reviewed contract table and explicit compatibility requirements before
any public declaration is moved.

### Phase 2: introduce the API foundation

- Add CUDA backend selection, operation capability declarations, and non-owning
  execution/memory descriptors without changing existing CPU calls.
- Extract shared enums with compatibility includes. Reuse `workspace_arena`.
- Define CUDA-disabled diagnostics and ensure all canonical headers can be
  included in CPU-only builds without CUDA headers or libraries.

Exit: existing consumers still compile, and unsupported backend combinations
produce intentional diagnostics.

### Phase 3: prove the design with multiplication and transpose

- Add canonical CUDA specializations that initially delegate to existing GPU
  functions, keeping kernels untouched.
- Validate both old and new entry points against the same inputs, including
  transposes, rectangular matrices, padded strides where supported, and streams.
- Review signature ergonomics and exported symbols before expanding the pattern.

Exit: a complete usable CPU/CUDA operation family with unchanged numerical and
execution behavior.

### Phase 4: migrate operations in dependency order

- Migrate factorizations, solves, and inversion, followed by SVD-based composite
  operations and reductions in the order listed above.
- Specify per-operation device info and pivot requirements rather than hiding
  them in host booleans. Preserve all existing scalar-return synchronization.
- Retain unsupported operations as explicit capability gaps.

Exit: every existing GPU operation has a canonical home and a tested legacy path.

### Phase 5: consolidate files and build integration

- Move CUDA implementations into `src/matrix_operation/cuda/` without changing
  numerical kernels. Make canonical implementations own the symbols, with legacy
  wrappers delegating to them; prevent circular forwarding or duplicate definitions.
- Update CMake discovery and backend filters. Its recursive `.cpp` glob currently
  includes GPU adapters, while `.cu` files are added only with CUDA enabled.
  Preserve equivalent guards or explicitly select CUDA sources after the move.
- Keep `LINALG_ENABLE_CUBLAS` and public `LINALG_HAS_*` definitions compatible.
  CPU-only configuration must never enable the CUDA language or link CUDA.
- Update Bazel header/source packaging deliberately. Its current GPU sources are
  excluded and it has no CUDA toggle; a unified directory must not accidentally
  pull CUDA adapters into the native glob. Adding Bazel CUDA execution support
  is outside this fusion unless separately scoped.
- Verify install/export rules include canonical and compatibility headers.

Exit: source, install, static/shared, and CPU-only builds preserve their supported
behavior with the consolidated directory structure.

### Phase 6: document and deprecate

- Update README examples and backend/GPU design documents to show canonical calls,
  memory domains, capabilities, workspace ownership, and blocking behavior.
- Keep old GPU include paths and exported function signatures as forwarding
  adapters for the agreed compatibility window. Header aliases alone do not
  preserve binary compatibility for already-linked consumers.
- Add deprecation notices only after replacements cover all existing use cases.
  Remove old headers/symbols only in an explicitly scheduled breaking release.

Exit: consumers have a complete migration guide, with removal timing documented.

## Validation strategy for implementation

Reuse the existing `Testing/Cxx/Test*.cxx` suites and
`TestGpuMatrixOperations.cxx`; add targeted coverage for new public contracts and
legacy adapters. This documentation-only task requires no compilation.

| Area | Required evidence when implementation begins |
| --- | --- |
| CPU portability | Native configuration with no CUDA toolkit; existing CPU tests pass |
| CPU vendors | Available BLAS/LAPACKE and MKL configurations retain explicit selection, including builds enabling both |
| CUDA | Real-device tests for float/double, default and non-default streams; no-device skips reported separately from passes |
| Mathematical correctness | Products, factor reconstructions, solve residuals, orthogonality, and singular/rank-deficient cases with precision-appropriate tolerances |
| Contracts | Supported strides, mutation, workspace boundaries, invalid dimensions, checked narrowing, and backend capability diagnostics |
| Asynchrony | Stream ordering and scratch lifetime exercised without adding device-wide synchronization inside adapters |
| Compatibility | Old/new headers and calls compile; old exported symbols remain; installed consumers link against static/shared libraries |
| Build integration | Supported Bazel CPU configuration remains clean; CUDA stays optional in CMake |
| Performance | Compare pilot and composite-operation baselines with identical inputs/streams; investigate extra allocations, copies, synchronization, or dispatch overhead |

Use the repository's setup helpers and existing CI configurations for actual
commands. Run GPU checks on a CUDA-capable runner; CPU-only results cannot
establish GPU correctness or timing.

## Risks and boundaries

- **API uniformity can conceal execution differences.** Keep address space,
  device status, workspace ownership, and blocking semantics visible in signatures
  and documentation.
- **Directory moves can alter build membership.** Audit CMake and Bazel source
  lists before and after each move, especially native versus CUDA `.cpp` files.
- **Compatibility wrappers can change ABI or overload resolution.** Preserve
  concrete exported legacy symbols and test both old and canonical call sites.
- **Shared code can impose unsupported contracts.** Keep packing, pivoting, and
  numerical limitations backend-specific until separately implemented and tested.
- **Scope can expand into algorithm work.** New GPU algorithms, automatic backend
  choice, allocation management, asynchronous scalar APIs, and broad index-type
  cleanup are follow-up projects rather than prerequisites for fusion.

## Completion criteria

The fusion is complete when every existing operation is accounted for under the
canonical family, backend support is explicit, CUDA sources share the operation
hierarchy, existing callers retain their supported behavior, and required build,
numerical, stream, packaging, and compatibility checks pass. Legacy forwarding
headers may remain until the scheduled breaking release.
