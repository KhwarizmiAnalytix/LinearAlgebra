# Sparse matrix integration: benefits and scope

Date: 2026-10-17. Status: documentation-only proposal; sparse support is
not implemented. The user confirmed that the requested topic is sparse
matrix operations. This proposal describes an intentional expansion beyond
the dense-only scope currently stated in the README.

Start with portable CSR storage, sparse matrix-vector multiplication, and
a preconditioned conjugate-gradient solver for symmetric positive-definite
systems. Keep the current dense and banded operations available. Add vendor
acceleration only after a real consumer demonstrates value.

Read this document for the rationale, the
[technical design](sparse_matrix_design.md) for contracts and architecture,
and the [implementation plan](sparse_matrix_implementation_plan.md) for
phases, acceptance criteria, testing, and rollout. All names introduced in
these documents are proposals unless explicitly identified as existing.

## 1. What sparse operations add

A sparse matrix has enough zero entries that recording the stored entries
and their positions is useful. Dense storage reserves space for every
entry, including every zero. Sparse storage carries index overhead but can
avoid most of those entries and the arithmetic associated with them.

Two useful quantities are density = nnz / (rows × columns) and sparsity =
1 − density. Here, nnz means the number of stored entries; explicitly
stored zeros count toward nnz even though their numerical value is zero.
Sparsity alone does not predict speed: row lengths, access patterns,
problem size, reuse, and conversion costs also matter.

| Operation | Meaning | Practical purpose |
|---|---|---|
| SpMV | Sparse matrix × dense vector | Residuals, iterative solves, graph propagation |
| SpMM | Sparse matrix × dense matrix | Several vectors or right-hand sides at once |
| Sparse transpose | Rearrange indexed entries | Transpose products and adjoint systems |
| Assembly | Combine coordinate contributions into compressed storage | Mesh elements, graph edges, sparse Jacobians |
| Iterative solve | Repeated products and vector updates | Large systems without dense factorization |
| Sparse direct solve | Ordering, analysis, factorization, solve | Robust factor reuse when memory permits |
| SpGEMM | Sparse matrix × sparse matrix | Operator composition, with potentially much denser output |

The MVP should cover assembly, SpMV, and the first iterative solve. SpMM
and transpose follow; general direct solvers and SpGEMM are later work.

## 2. What this repository already provides

These findings come from the checked-out source, not from assuming the
existing roadmap has been fully implemented or remains current.

| Existing area | Evidence | Consequence for sparse integration |
|---|---|---|
| Dense CPU kernels with explicit backend templates | [backend.h](../include/common/backend.h), [multiplication.h](../include/matrix_operation/multiplication.h) | Add separate sparse operations; retain existing dense contracts |
| Shared signed 64-bit public index type | [types.h](../include/common/types.h) | Use it for dimensions; design compressed index width explicitly |
| Caller-owned scratch planning | [workspace_arena.h](../include/common/workspace_arena.h) | Query storage requirements and accept caller buffers |
| Checked vendor integer conversions | [narrow.h](../include/common/narrow.h) | Extend range checks to sparse offsets, indices, and capacities |
| Project error macros | [linalg_checks.h](../include/common/linalg_checks.h) | Follow existing error conventions |
| Dense CUDA operations with streams | [GPU multiplication header](../include/matrix_operation_gpu/multiplication_gpu.h) | Introduce an independent cuSPARSE adapter with explicit device storage |
| Specialized banded operations | [tridiagonal header](../include/matrix_operation/tridiagonal_operations.h), [pentadiagonal header](../include/matrix_operation/pentadiagonal_operations.h) | Retain efficient paths for fixed narrow bands |
| CMake and Bazel integration | [CMakeLists.txt](../CMakeLists.txt), [BUILD.bazel](../BUILD.bazel) | Register new trees in both systems; existing Bazel globs omit them |
| Backend comparison test helper and benchmarks | [test_backend.h](../Testing/Cxx/test_backend.h), [benchmark configuration](../Testing/Benchmark/CMakeLists.txt) | Extend established verification infrastructure |

No general CSR/CSC/COO API or SpMV/SpMM implementation was found in the
first-party headers and sources inspected. The README explicitly describes
sparse backends as outside the current scope.

The [existing backend roadmap](backend_roadmap.md) is historical context:
its description of free functions, internal scratch allocation, absent
benchmarks, and missing backend CI coverage differs from current headers,
benchmark files, and [CI configuration](../.github/workflows/ci.yml).
Current CI defines BLAS, pivoting, sanitizer, and CUDA compile-only jobs as
well as other configurations. A configured job is not evidence that its
latest run passed. This documentation task did not execute project builds.

## 3. Quantified storage benefits

For an m × n matrix, value width v bytes, and compressed index width i
bytes, the simple storage model is:

- Dense bytes = m × n × v.
- CSR bytes = nnz × (v + i) + (m + 1) × i.
- Ignoring the row-pointer term, the memory break-even density is v / (v + i).

For doubles and 32-bit indices, that asymptotic memory threshold is about
66.7%; for doubles and 64-bit indices it is 50%. These are storage
thresholds, not performance thresholds or recommendations to switch at
those densities.

| Illustrative matrix | nnz | Dense doubles | CSR, 32-bit indices | Dense / CSR storage |
|---|---:|---:|---:|---:|
| 1,000 × 1,000, 50% stored | 500,000 | 8 MB | 6.004 MB | 1.33× |
| 10,000 × 10,000, 0.1% stored | 100,000 | 800 MB | 1.240 MB | 645× |
| 100,000 × 100,000, average 7 stored per row | 700,000 | 80 GB | 8.800 MB | 9,091× |

These are calculations, not measurements. MB and GB are decimal. They
exclude vectors, alignment, assembly workspace, converted indices, vendor
metadata, factors, and solver workspaces. The last CSR example takes about
12 MB with 64-bit indices. Dense conversion of that example still requires
80 GB just for matrix values.

Dense matrix-vector multiplication visits O(mn) entries; CSR SpMV visits
O(m + nnz) data, including row handling. For fixed average row width, CSR
matrix storage grows linearly with row count. A sparse iterative solve
still depends on iteration count and preconditioning: replacing storage
does not guarantee a faster solve or convergence.

## 4. How this can benefit your projects

Only this repository was reviewed. Related project relationships below
come from its documentation; their source and workloads were not audited.
Domain use cases are opportunities to validate, not claims that those
applications already exist in your projects.

| Project or workload | Proposed integration | Expected benefit | Evidence needed |
|---|---|---|---|
| LinearAlgebra | Separate sparse kernels alongside dense and banded kernels | Larger feasible systems and a reusable computational core | Real sparse matrices and end-to-end comparisons |
| XSigma / Hisab consumer mentioned in the README | Assemble operator contributions directly into COO, then CSR | Avoid allocating a dense global operator | Confirm which consumer operators are sparse and where assembly occurs |
| Separate Memory library referenced by the roadmap | Allocate values, indices, vectors, and scratch in the consumer | Reuse existing allocation strategy without a mandatory dependency | Check alignment, lifetimes, host/device placement, and peak memory |
| Solver layer or other consumer | Use SpMV and convergence reports to build numerical algorithms | Separate domain modeling from storage and arithmetic | Confirm SPD/general structure, tolerances, and repeated-solve patterns |
| Profiler integration point in CMake | Measure assembly, setup, products, iterations, transfers | Explain whether kernel speed becomes application speed | Extend available profiling coverage to the future sparse operations |
| PDE / finite element or finite difference applications | Sparse discretized operators and preconditioned solves | Local couplings can remain compact as problems grow | Boundary conditions, nullspaces, and conditioning |
| Graph and network applications | Weighted adjacency or Laplacian products | Work scales with stored edges rather than all possible edges | Degree distribution; singular graph Laplacians require constraints |
| Optimization and calibration | Sparse Jacobian products and later LSQR | Avoid dense rectangular matrices and explicit pseudo-inverses | Jacobian sparsity, rank, scaling, and required solution quality |
| Adjoint / sensitivity applications | Later transpose solves and gradients over stored entries | Reuse operator structure during sensitivity calculations | A separate derivative design; sparse support does not provide AAD automatically |

For repeated solves, distinguish three cases: unchanged matrix with new
right-hand sides; unchanged pattern with new values; and a changed
pattern. They permit different amounts of preprocessing, preconditioner,
or factor reuse. Incorrect reuse can invalidate results.

## 5. Choosing the right representation

| Situation | Preferred starting point | Reason |
|---|---|---|
| Small dense block or mostly populated matrix | Existing dense operations | Contiguous storage and optimized dense kernels |
| Strict tridiagonal or pentadiagonal system | Existing banded operations | Avoid generic sparse index overhead |
| Irregular sparse operator, repeated row products | CSR | Natural row traversal and broad interoperability |
| Unsorted element or edge contributions | COO during assembly, then CSR | Contributions can be sorted and duplicates combined |
| Column-oriented external solver | CSC adapter when needed | Avoid forcing the solver to consume a mismatched layout |
| Repeated dense blocks on graph edges | Later BSR evaluation | Block reuse may justify a block sparse representation |
| Operator cheaply evaluated without stored entries | Later matrix-free interface | May avoid even compressed matrix storage |

Do not replace all dense matrices, automatically scan dense arrays on every
call, or convert existing narrow-band systems to CSR by default. Assemble
sparse storage at the source of the operator when possible.

Avoid treating inverse, full SVD, full eigendecomposition, and the full
matrix exponential as simple sparse equivalents of the current dense
operations. Their outputs or intermediate factors can become dense. For
large sparse systems, prefer solving for the required vectors; later
consider selected eigenpairs or the action exp(tA)b. The latter has a
well-established interface illustrated by
[SciPy's exponential-action documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html).

## 6. Recommended investment and success conditions

Build the native CPU foundation first. Add CG with a diagonal
preconditioner only for symmetric positive-definite systems. Measure one
real consumer before expanding to general solvers or accelerators.

Adopt sparse support when the full workflow meets the consumer's residual
and accuracy requirements and either reduces peak memory enough to enable
a needed problem size or improves total runtime by an agreed useful
margin. Track assembly, conversion, setup, and transfers in that decision.
Keep dense or banded paths when they perform better.

The main costs are a second storage contract, index validation, assembly
and pattern lifetimes, convergence diagnostics, preconditioner work, and
backend verification. Budget them as a new capability rather than a small
change to dense multiplication. The following documents make that work
reviewable without implementing it:

- [Technical design and proposed contracts](sparse_matrix_design.md).
- [Phased delivery, estimates, validation, and rollout](sparse_matrix_implementation_plan.md).
