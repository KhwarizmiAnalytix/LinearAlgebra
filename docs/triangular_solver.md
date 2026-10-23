# Triangular solver

`linalg::triangular_solver<T, Backend>` solves `op(A) * X = B`. `A` is a
nonsingular real triangular matrix; `B` and `X` have one column per right-hand
side. All matrices are row-major and may have padded row strides. `A`, `B`, and
`X` use distinct caller-owned buffers; the inputs and output padding are
preserved. Singular matrices are not diagnosed.

```cpp
using Solver = linalg::triangular_solver<double, linalg::linalg_backend::NATIVE>;
const auto bytes = Solver::workspace_size(n, nrhs);
// Allocate caller-owned scratch with at least bytes bytes and double alignment.
Solver::apply(/*upper=*/true, /*transpose=*/false, /*unit_diagonal=*/false,
    n, nrhs, A, lda, B, ldb, X, ldx, workspace, bytes);
```

Only the selected triangle is read. With `unit_diagonal=true`, diagonal
entries are implicit ones and are not read. With `transpose=true`, the
selected stored triangle is transposed before solving.

The native implementation uses 32-row blocks, SIMD across independent RHS
columns, and `multiplication<T, NATIVE>` for updates outside each diagonal
block. It falls back to scalar substitution when vectorization is disabled.
BLAS and MKL call row-major TRSM. CUDA and HIP preserve `B` with a device copy,
then call TRSM on the supplied stream using the equivalent column-major
right-side solve. Metal uses `MPSMatrixSolveTriangular` with separate RHS and
solution buffers.

CPU backends support `float` and `double` host buffers; CUDA/HIP support both
types with device buffers. Metal supports `float` with page-aligned shared
buffers and follows the existing synchronous queue contract. Workspace size
is backend-specific: query it rather than assuming it is zero.

The matrix logarithm uses this API for its Gauss-Legendre quadrature systems;
its complex unit-triangular systems are embedded into equivalent real systems.
