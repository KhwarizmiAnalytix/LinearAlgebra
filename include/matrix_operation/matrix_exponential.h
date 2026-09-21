#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define linalg_int __int64
#define linalg_long unsigned __int64
#else
#define linalg_int long long int
#define linalg_long unsigned long long int
#endif

namespace linalg
{

// Host-memory-only CPU entry point. Computes result = exp(A) for a square,
// row-major n x n matrix A (lda/ldc >= n) via the scaling-and-squaring
// method with diagonal Pade approximants of degree 3/5/7/9/13, selected from
// ||A||_1 against the theta thresholds of N. J. Higham, "The Scaling and
// Squaring Method for the Matrix Exponential Revisited", SIAM J. Matrix
// Anal. Appl. 26(4), 2005 — the algorithm PyTorch's torch.matrix_exp /
// torch.linalg.matrix_exp and MATLAB's expm are built on. A is read-only;
// result may safely alias A.
//
// Composed entirely on top of linalg::matrix_multiplication,
// linalg::matrix_invert, and linalg::matrix_norm (ONE) — there is no
// separate backend selection here, this op simply inherits whichever CPU
// backend (scalar / blas_lapack / mkl) those were built with. No GPU
// implementation exists for this op.
LINALG_API void matrix_exponential(const float* A, linalg_int n, linalg_int lda, float* result, linalg_int ldc);

LINALG_API void matrix_exponential(const double* A, linalg_int n, linalg_int lda, double* result, linalg_int ldc);

}  // namespace linalg
