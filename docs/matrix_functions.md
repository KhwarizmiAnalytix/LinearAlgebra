# Matrix logarithm and square root

`linalg::logarithm<T, Backend>` computes the principal matrix logarithm;
`linalg::square_root<T, Backend>` computes the principal matrix square root.
These are matrix functions, with `exp(log(A)) = A` and `sqrt(A) * sqrt(A) = A`
to numerical precision. They accept general real square matrices, including
nonsymmetric matrices, conjugate eigenvalue pairs, and defective matrices.

Both operations expose:

```cpp
using Operation = linalg::logarithm<double, linalg::linalg_backend::NATIVE>;
const size_t bytes = Operation::workspace_size(n);
// Allocate caller-owned scratch with at least bytes bytes and double alignment.
Operation::apply(A, n, lda, result, ldc, workspace, bytes);
```

Matrices are row-major with `lda >= n` and `ldc >= n`. The input is preserved
unless the output aliases it; exact in-place operation is supported. Matrix
order must be positive, and orders and row-stride spans must fit the core's
32-bit indexing. Query `workspace_size(n)` for the backend's exact required
size; it is not a fixed formula, since it folds in the dedicated
`schur_decomposition<T, Backend>` kernel's own workspace, `multiplication<T,
Backend>`'s GEMM workspace (reconstruction), and, for logarithm,
`triangular_solver<T, Backend>`'s workspace (the seven quadrature solves) —
each backend-dependent. Beyond those, the plan itself holds eight real n x n
matrices for reconstruction plus `schur_decomposition`'s own Q/S output
planes, and, for logarithm, a further six real n x n matrices, one complex
n-vector, and three state scalars. Metal aligns each sub-buffer to the system
page size. Logarithm orders must also allow its `2*n` real system to fit
32-bit indexing. No numerical buffers are allocated by the library.

## Domains and failures

The real principal logarithm requires no eigenvalues on the nonpositive
real axis. The real principal square root requires no negative real
eigenvalues. Zero eigenvalues are supported when the triangular recurrence
is defined, including diagonal positive semidefinite matrices. Singular
nontrivial Jordan blocks can have no square root and are rejected.

The public scalar types are `float` and `double`; this library does not
currently expose TensorFlow's complex-valued logarithm API. Nonfinite
inputs and nonfinite results are treated as numerical failures. Schur
iteration is bounded, and a material imaginary residual is rejected.
Floating-point computation near the branch cut or a singular matrix can
fail even when an exact mathematical function exists.

CPU numerical or domain failures throw using the existing Logging checks.
CUDA and ROCm enqueue on the supplied stream and write NaNs on numerical
or domain failure, without downloading a status or synchronizing. Metal
also writes NaNs and follows the existing synchronous command-queue API.
Invalid host launch arguments, such as insufficient workspace, throw on
every backend.

## Backends and performance

| Backend | Types | Matrix and workspace memory |
|---|---|---|
| NATIVE | float, double | Host |
| BLAS_LAPACKE | float, double | Host; LAPACKE is not required for these two operations |
| MKL | float, double | Host |
| CUDA_BLAS | float, double | Device |
| ROCM_BLAS | float, double | Device |
| METAL_BLAS | float | Page-aligned shared buffers, following `metal_handle.h` |

Each operation has its own implementation file: `cpu/logarithm.cpp` and
`cpu/square_root.cpp`, with matching `_cublas.cu`, `_hip.hip`, and `_metal.mm`
files under the GPU backend directories. Each source owns its operation's
implementation and launch code; there are no `matrix_functions*` implementation
files. Both call the dedicated `schur_decomposition<T, Backend>` kernel
declared in [`matrix_operation/schur_decomposition.h`](../include/matrix_operation/schur_decomposition.h)
— the same kernel alongside `lu_decomposition.h`/`qr_decomposition.h`/
`cholesky_decomposition.h`/`svd_decomposition.h` — to factor `A = Q*S*Q^*`
and then operate only on the triangular `S` it returns; neither operation
embeds its own copy of the Schur reduction. The square-root and logarithm
numerical helpers are otherwise separate, and logarithm reuses the triangular
square-root recurrence. This preserves support for defective matrices that
cannot be evaluated through an eigenvector decomposition.

Reconstruction of `Q*f(T)*Q^*` uses `multiplication<T, Backend>` with real
and imaginary factors. Each complex product uses four real GEMMs, reusing
the native SIMD, BLAS, MKL, cuBLAS, hipBLAS, or Metal MPS implementation.
On CPU, the Schur Householder updates, Givens rotations, and triangular
square-root inner products reuse the existing `packet<T>` SIMD abstraction
when vectorization is enabled. A scalar implementation remains available.
Recurrence dependencies and convergence decisions retain their required order.

The logarithm delegates each of its seven quadrature systems to
`triangular_solver<T, Backend>`. Each complex row is divided by its diagonal,
then real and imaginary components are interleaved into a real unit-upper
triangular system of order `2*n` with `n` right-hand sides. The native solver
uses SIMD substitution within blocks and existing GEMM for block updates.
Vendor backends use BLAS/MKL/cuBLAS/hipBLAS TRSM; Metal uses MPS's triangular
solver. No LU factorization or host transfer is needed for these systems.

The GPU Schur reduction (inside `schur_decomposition<T, Backend>`) and the
triangular square-root/logarithm recurrences still run on one device thread;
the delegated GEMMs and triangular solves use the backend kernels. Numerical
data and scratch stay on the device. Work is cubic per Schur sweep and
reconstruction; iteration counts depend on the spectrum. CUDA/HIP enqueue the
Schur decomposition, preparation, triangular solves, GEMMs, elementwise
combinations, and validation on the caller's stream without synchronizing.
Metal's `schur_decomposition` call and each subsequent stage commit and wait
on their own, since Metal has no caller-continuable stream handle.

## TensorFlow and Eigen provenance

TensorFlow v2.20.0's
[MatrixLogarithm](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/core/kernels/linalg/matrix_logarithm_op.cc)
and
[MatrixSquareRoot](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/core/kernels/linalg/matrix_square_root_op.cc)
copy a row-major matrix and delegate to Eigen's `MatrixFunctions` module.
They are CPU kernels, so they cannot supply CUDA/ROCm/Metal implementations
directly.

The [square-root helper](../src/common/square_root_core.h) and
[logarithm helper](../src/common/logarithm_core.h) adapt the triangular
square-root recurrence and inverse scaling/Gauss-Legendre logarithm
evaluation from Eigen's
[MatrixSquareRoot.h](https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/src/MatrixFunctions/MatrixSquareRoot.h)
and
[MatrixLogarithm.h](https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/src/MatrixFunctions/MatrixLogarithm.h).
The adaptation uses an allocation-free complex Hessenberg/Schur reduction,
a fixed degree-seven logarithm approximation with a conservative scaling
threshold, cancellation-safe diagonal evaluation, and shared device code.
It introduces no TensorFlow or Eigen build/runtime dependency.

Eigen copyright notices and its MPL 2.0 file-level license are retained in
the adapted helpers; the license text is in [LICENSES/MPL-2.0.txt](../LICENSES/MPL-2.0.txt).
