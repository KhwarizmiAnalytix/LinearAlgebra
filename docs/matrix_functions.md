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
32-bit indexing. Workspace contains five complex matrices and one complex
vector: `(10*n*n + 2*n) * sizeof(T)` bytes. No numerical buffers are allocated
by the library.

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

All backends use the same portable complex Schur implementation. CPU
specializations share the scalar core; they do not call vendor GEMM or
Schur routines. Each GPU call executes the core on one device thread and
keeps numerical data and scratch on the device. This establishes backend
coverage and stream behavior, but is not optimized for large matrices or
GPU throughput. The work is cubic per Schur sweep and reconstruction;
iteration counts depend on the spectrum.

## TensorFlow and Eigen provenance

TensorFlow v2.20.0's
[MatrixLogarithm](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/core/kernels/linalg/matrix_logarithm_op.cc)
and
[MatrixSquareRoot](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/core/kernels/linalg/matrix_square_root_op.cc)
copy a row-major matrix and delegate to Eigen's `MatrixFunctions` module.
They are CPU kernels, so they cannot supply CUDA/ROCm/Metal implementations
directly.

The shared [numerical core](../src/common/matrix_functions_core.h) adapts the
triangular square-root recurrence and inverse scaling/Gauss-Legendre
logarithm evaluation from Eigen's
[MatrixSquareRoot.h](https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/src/MatrixFunctions/MatrixSquareRoot.h)
and
[MatrixLogarithm.h](https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/src/MatrixFunctions/MatrixLogarithm.h).
The adaptation uses an allocation-free complex Hessenberg/Schur reduction,
a fixed degree-seven logarithm approximation with a conservative scaling
threshold, cancellation-safe diagonal evaluation, and shared device code.
It introduces no TensorFlow or Eigen build/runtime dependency.

Eigen copyright notices and its MPL 2.0 file-level license are retained in
the adapted core; the license text is in [LICENSES/MPL-2.0.txt](../LICENSES/MPL-2.0.txt).
