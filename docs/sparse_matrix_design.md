# Sparse matrix technical design

Date: 2026-10-17. Status: proposed architecture only. No APIs, build flags,
files, or algorithms described as proposed have been implemented.

See the [benefits and scope](sparse_matrix_integration.md) and
[implementation plan](sparse_matrix_implementation_plan.md).

## 1. Architecture and boundaries

Keep LinearAlgebra a kernel library. Consumers own matrices, vectors,
assembly buffers, and workspaces; allocation can come from the separate
Memory project or any other allocator. Add a sparse module without
changing existing dense signatures or forcing dense consumers to link a
new vendor library.

| Layer | Responsibility | Ownership |
|---|---|---|
| Consumer / application | Mesh, graph, model, contributions, constraints, tolerances | Domain objects and all public numerical buffers |
| Sparse views | Describe dimensions, compressed buffers, capacities, and types | Borrowed storage only |
| Assembly and native kernels | Validate, canonicalize, multiply, transpose | Caller-provided outputs and scratch |
| Solver layer | Iteration, preconditioning, stopping, diagnostics | Caller-provided vectors and preconditioner data |
| Optional adapter plans | Vendor handles, analysis, supported capabilities | Explicit lifecycle; backend-specific internal allocations disclosed |

An initial namespace of `linalg::sparse` is proposed for CPU operations and
`linalg::gpu::sparse` for CUDA operations. Public views are small borrowed
descriptors, not owning matrix classes. No container framework or mandatory
Eigen dependency is needed.

## 2. Storage types and invariants

### 2.1 CSR as the first execution format

A proposed CSR view describes m rows, n columns, nnz stored entries,
values, column indices, and row offsets. It also carries buffer lengths
so the library can check declared capacity. Raw pointers cannot prove
that the caller actually allocated the claimed storage; this remains a
caller obligation.

| Buffer | Logical length | Meaning |
|---|---:|---|
| Values | nnz | Stored coefficients |
| Column indices | nnz | Column of each coefficient |
| Row offsets | m + 1 | Start and end positions for each row |

Use zero-based indexing throughout the public sparse API. A valid canonical
CSR matrix has row_offsets[0] = 0, row_offsets[m] = nnz, nondecreasing row
offsets, and column indices in [0, n). Within each row, columns are strictly
increasing; duplicate coordinates have already been combined. Equal
successive row offsets represent empty rows.

For example, a 3 × 4 matrix with rows (2, 0, 0, 5), (0, 0, 0, 0), and
(0, 7, 0, 0) has:

- Values: [2, 5, 7].
- Column indices: [0, 3, 1].
- Row offsets: [0, 2, 2, 3].

An absent entry means zero. Explicitly stored zeros are valid and retained
by default, which preserves a reusable pattern. The initial compute
contract requires the full matrix to be stored, even for symmetric
matrices; storing one triangle and treating it as the full matrix requires
a later explicit symmetry contract.

### 2.2 Scalar and index support

Start with float and double coefficients and signed 32-bit or 64-bit
compressed indices. Use the same index width for row offsets and column
indices in the initial native format. Dimensions and counts at the public
boundary use existing `linalg::index_t`, a signed 64-bit type. Reject
negative dimensions, counts, and indices before converting to unsigned
byte counts.

Check m + 1, nnz, leading dimensions, capacity products, and every byte
calculation for overflow. A 32-bit representation must fit its offsets,
indices, and dimensions within the documented supported range. Converting
to a vendor type requires checked value conversion, not pointer casts.
Even equal-width integer types need compatible actual C++ types or an
explicitly converted buffer before their arrays can be passed safely.

Use existing [narrowing helpers](../include/common/narrow.h) where their
contract applies; add checked arithmetic in the future implementation
where multiplication and addition need protection. Complex coefficients
and mixed-width offset/index layouts are deferred.

### 2.3 COO assembly and canonicalization

Accept COO contributions as row indices, column indices, and values.
Contributions can arrive unsorted and contain repeated coordinates, as
happens when adjacent elements contribute to the same matrix entry.

The proposed assembly sequence is: validate bounds, sort by row and column,
sum duplicates in a documented order, optionally prune, count rows, and
emit canonical CSR. Report the actual stored count. The default preserves
explicit zeros and zeros produced by cancellation; an explicit pruning
option changes the pattern. Removing values below a tolerance changes the
numerical operator and must require an explicit consumer choice.

Before canonicalization, input contribution count is an upper bound on
output nnz. Let callers query scratch and worst-case output capacities
from that count, then receive the actual output count. A later exact-size
two-pass mode can avoid excess capacity at the cost of another pass.
Native assembly must not allocate internally to discover output size.

Stable duplicate accumulation can make a given input order reproducible;
floating-point sums need not match after contributions are reordered.
Sorted compression and duplicate merging are standard sparse assembly
concepts; [Eigen's sparse tutorial](https://eigen.tuxfamily.org/dox/group__TutorialSparse.html)
provides a reference. The buffer-based ownership choices here are specific
to this proposal.

### 2.4 Validation and empty matrices

Separate cheap per-call checks from full O(m + nnz) structural validation.
Public calls validate dimensions, capacities, pointers, and workspace;
full validation occurs at assembly/import or explicit plan creation. A
validated borrowed view is only valid while its structural arrays remain
unchanged. Provide an explicit checked path for untrusted imported data;
do not silently trust arbitrary user arrays.

Allow zero-sized shapes and nnz = 0. Row offsets still contain m + 1
entries, all zero for an empty pattern. Values and column-index pointers
may be null when nnz = 0. Nonempty logical dense inputs or outputs still
require storage. Cover m = 0, n = 0, and nonzero dimensions with zero nnz
separately. Sparse empty-input semantics need not alter existing dense
operations, which currently reject some zero dimensions.

Reject structural or workspace errors through existing `LINALG_CHECK` /
`LINALG_THROW` conventions. Distinguish unsupported backend capabilities
from malformed storage and numerical solver termination.

## 3. Operation contracts

| Proposed operation | Contract | Storage or execution requirement |
|---|---|---|
| CSR SpMV | y = alpha × op(A) × x + beta × y | First release supports op(A) = A; explicit transpose later |
| CSR SpMM | C = alpha × op(A) × B + beta × C | Dense B/C are row-major with explicit leading dimensions |
| CSR transpose / CSR-to-CSC | Produce a new compressed representation | Query output and scratch capacities; no implicit mutation |
| Diagonal extraction | Produce diagonal and missing-entry information | Missing or zero diagonals need a preconditioner policy |
| COO-to-CSR | Canonicalize contributions into caller outputs | Return actual nnz and preserve inputs |
| Dense-to-CSR | Explicit opt-in import | Scan cost O(mn); no automatic per-call conversion |
| CSR-to-dense | Explicit opt-in export | Check dense capacity and byte overflow before writing |

For SpMV, an m × n matrix needs x of length n and y of length m; transpose
reverses those lengths. For SpMM, op(A) is m × k, B is k × r, and C is
m × r. Require leading dimensions at least r for the initial untransposed
row-major B/C layout. Restrict initial dense operand transpose support to
documented cases rather than copying every dense GEMM option immediately.

No input/output overlap is permitted for these first sparse products.
In-place x = A × x requires an explicit temporary chosen by the caller.
Workspace must not overlap any live operand. When beta = 0, kernels must
not read old output values; when alpha = 0, only the beta scaling is
numerically needed, while required shape and structural validation still
applies. Specify and test NaN/infinity behavior rather than assuming
reordering or zero skipping is equivalent to dense IEEE arithmetic.

The native baseline should be serial and deterministic for a fixed
canonical input and build. Parallel reductions may change rounding.
Compare backends using tolerances; offer bitwise reproducibility only
where a tested implementation explicitly supports it.

## 4. Workspace and reusable plans

Expose byte requirements for temporary storage and any persistent caller
buffers. Queries may depend on shape, nnz, index type, operation, backend,
algorithm, and sometimes the actual pattern. Do not promise a shape-only
query for a backend that needs matrix descriptors or analysis.

Reuse the idea of [workspace_arena](../include/common/workspace_arena.h),
but validate required base alignment and arithmetic overflow explicitly.
Aligned offsets alone cannot fix a misaligned base pointer. Native kernels
must not perform hidden heap allocation. A zero-byte requirement permits
a null workspace; otherwise reject null, misaligned, or undersized storage.

An optional explicit plan follows this lifecycle:

1. Validate/import pattern and declare operation and backend.
2. Query temporary and persistent requirements.
3. Supply caller buffers and construct backend metadata.
4. Analyze or preprocess if worthwhile.
5. Execute repeatedly while the declared inputs remain valid.
6. Complete queued work before updating or releasing plan resources.
7. Destroy vendor metadata and release caller storage in the right order.

Define invalidation precisely:

| Change | Required action |
|---|---|
| New right-hand side or output | Reuse only when layout, strides, and plan contract permit |
| New coefficients, same pattern | Update value binding; refresh value-dependent preconditioners or factors; check vendor preprocessing rules |
| Changed row offsets or columns | Revalidate and rebuild structural analysis |
| New dimensions, nnz, index width, transpose, or algorithm | Recreate affected plan |
| New CUDA device, execution context, or workspace | Rebind or rebuild as explicitly supported |

Associate reusable analysis with a caller-maintained pattern identity and
revision, plus shape, layout, index width, backend, and operation. Value
revisions identify factors and preconditioners separately. Pointer identity
alone is insufficient because storage can be edited in place. Avoid
content hashing every execution, which would add another O(nnz) scan.

Vendor metadata is an important exception to a literal no-allocation rule:
opaque libraries may allocate internally during handle creation or
analysis. Keep the native path strictly caller-buffer based; document
vendor-managed metadata allocations and expose their lifecycle. If a
consumer requires a strict total allocation bound that the vendor cannot
provide, use the native path or report that backend as unsuitable. Do not
describe vendor internals as caller-controlled memory.

## 5. Solver strategy and numerical behavior

### 5.1 First solver: preconditioned CG

CG applies to real symmetric positive-definite systems. Symmetry alone
does not establish positive definiteness; callers declare this property,
tests verify it for fixtures, and runtime breakdown checks catch some
violations. A full SPD proof is not part of the cheap input validation.

Start with identity and positive-diagonal Jacobi preconditioners. Reject
invalid, missing, nonfinite, or nonpositive Jacobi diagonals rather than
silently inventing a replacement. Preconditioning must preserve the
requirements of the algorithm. [Eigen's sparse solver guide](https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html)
is a reference for matching solver families to matrix properties.

Accept an initial guess, maximum iterations, absolute tolerance, and
relative tolerance. Recompute the true residual r = b − A × x before
reporting success and use the proposed condition:

||r||₂ ≤ max(absolute_tolerance, relative_tolerance × ||b||₂).

The absolute tolerance makes b = 0 meaningful. Tolerances must be finite
and nonnegative. The result records termination reason, iterations, true
residual norm, and the threshold used. Expected outcomes include converged,
iteration limit, numerical breakdown, nonfinite arithmetic, and detected
nonpositive curvature. Invalid buffers remain API errors; failure to
converge is a numerical result, not a fabricated solution.

A CG iteration needs a sparse product, vector updates, dot products, and
preconditioner application. Storage is O(n) beyond A for the baseline,
but the exact vector count and byte requirements must come from the future
implementation's shared workspace plan. Periodic true-residual refresh
can limit drift; its cadence is an explicit performance/accuracy choice.

### 5.2 Later solvers

| Matrix / problem | Candidate | Additional work |
|---|---|---|
| Symmetric indefinite | MINRES | Symmetry-preserving preconditioning and breakdown contracts |
| General square | BiCGSTAB or restarted GMRES | Breakdown handling; GMRES restart controls and O(n × restart) basis storage |
| Rectangular least squares | LSQR / LSMR | A and transpose-A products, scaling, and least-squares stopping criteria |
| Repeated sparse direct solves | Vendor or SuiteSparse factorization | Ordering, symbolic/numeric separation, fill-in, factors, permutations |

LSQR can work from A × x and Aᵀ × x without forming an explicit
pseudo-inverse, as documented in the
[SciPy LSQR reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html).
For this project, prefer such an approach over automatically forming AᵀA:
normal equations can worsen conditioning and create additional fill.

Sparse LU/Cholesky must not be implemented by passing compressed arrays
to the current dense solver or by densifying invisibly. Reordering can
reduce fill, but sparse factors may still be much larger than A. Direct
solvers need separate memory reporting and validation gates.

## 6. Backend selection

Keep sparse capability selection separate from the existing dense enum:
the `blas_lapacke` choice does not establish a sparse API. Proposed sparse
CPU choices are native, MKL Sparse, and later an Accelerate-specific
adapter. CUDA uses a separate cuSPARSE path. Requesting an unsupported
operation/index combination must produce a clear capability error; any
fallback must be explicit and must not move data between host and device.

| Backend | Proposed role | Integration constraint |
|---|---|---|
| Native CPU | Portable baseline and correctness reference | Mandatory when sparse support is enabled |
| oneMKL Sparse | Repeated CPU sparse products and optional later solvers | Separate sparse capability checks and explicit handle lifecycle |
| cuSPARSE | Device-resident sparse products | Separate linkage, descriptors, streams, and workspace |
| Apple Accelerate Sparse | Optional macOS-specific adapter | Its sparse solvers use CSC; conversion and ownership need a separate design |
| SuiteSparse | Optional later sparse direct solver families | Select components and verify packaging and allocation needs |

Intel's inspector-executor interface separates analysis from execution and
supports several sparse formats. Prefer that supported interface for the
adapter rather than legacy Sparse BLAS.
[Intel Sparse BLAS reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2026-0/sparse-blas-routines.html).

The current project pins MKL ILP64 in CMake and asserts a 64-bit MKL index
width in [context.cpp](../src/common/context.cpp). Consequently, 32-bit CSR
indices require explicit conversion for that configuration. Allocate and
retain converted arrays at plan creation; do not reinterpret their
pointers. Intel's CSR creation API takes `MKL_INT` arrays, and its ordering
routine can modify indices: do not call it on borrowed immutable storage.
[Intel CSR handle documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/mkl-sparse-create-csr.html).

Apple's Sparse Solvers use CSC. Treat Accelerate sparse support as a
dedicated adapter rather than assuming the existing CBLAS option already
provides it. [Apple sparse matrix documentation](https://developer.apple.com/documentation/accelerate/creating-sparse-matrices).

SuiteSparse includes CHOLMOD for sparse Cholesky and UMFPACK for sparse LU.
Its packages have individual dependency and license information. Component
selection and compatibility with this project's distribution model belong
in the later dependency decision.
[Official SuiteSparse repository](https://github.com/DrTimothyAldenDavis/SuiteSparse).

These are candidate integrations; no vendor dependency is added by this
proposal. Verify the chosen minimum version, operation/type combinations,
and supported platforms when implementation starts.

## 7. CUDA-specific design

Use cuSPARSE Generic APIs for the proposed sparse products. Its documented
interfaces expose SpMV/SpMM workspace queries, external device buffers,
multiple index widths, and optional preprocessing. Exact support depends
on operation and algorithm.
[NVIDIA cuSPARSE reference](https://docs.nvidia.com/cuda/cusparse/index.html).

Keep values, offsets, column indices, vectors, and numerical scratch on
the device for execution. Host-side descriptors borrow that storage.
Extend the concept of the existing per-thread/per-device handle helpers
with a cuSPARSE handle and explicit stream binding; do not reuse cuBLAS
or cuSOLVER handles as sparse handles.

Plans record device, operation, algorithm, buffer bindings, and relevant
revisions. Treat preprocessing state as mutable and non-shareable across
concurrent executions unless the selected vendor contract explicitly
permits it. Independent streams need independent mutable plans and
workspaces. Test reusing one thread's context across different devices.

Input buffers, descriptors, analysis state, and workspace remain alive
until stream completion. Destruction must not race queued kernels. Do not
hide matrix uploads, downloads, or device-wide synchronization in product
kernels. Explicit import/validation may transfer structural metadata;
document that setup cost separately.

A first GPU iterative solver may synchronize at convergence checks to
return a host report. Expose this behavior and measure its cost. A later
asynchronous variant could return device-resident diagnostics for explicit
download. A stream argument alone does not make a whole solver nonblocking.

## 8. Deferred scope and design decisions

SpGEMM needs symbolic discovery of output structure before numeric output
can be sized. Sparse addition can also change the pattern. Both need
explicit output-capacity contracts and should follow the core release.

BSR, matrix-free operators, incomplete factorizations, algebraic multigrid,
distributed matrices, GPU direct solvers, full sparse AAD, partial
eigensolvers, and exponential actions are separate milestones. Do not
advertise them as consequences of implementing CSR SpMV.

Resolve these decisions at the indicated gates: exact public naming and
ABI before the foundation; mandatory 32/64-bit combinations before tests;
preprocessing lifetime before vendor adapters; and consumer tolerances and
memory budgets before claiming application benefits. The
[delivery plan](sparse_matrix_implementation_plan.md) assigns those gates.
