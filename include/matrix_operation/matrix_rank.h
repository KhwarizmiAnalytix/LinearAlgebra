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

// Host-memory-only CPU entry points, both routing through
// linalg::svd_decomposition (see svd_decomposition.h), which is itself the
// backend-dispatched (scalar / blas_lapack / mkl) primitive — there is no
// separate backend selection here. A is read-only in both. No GPU
// implementation exists for either op.

// Numerical rank: count of singular values > tol. tol < 0 (the default)
// picks tol = eps(T) * max(rows, columns) * largest singular value, matching
// the common LAPACK/NumPy rcond convention.
LINALG_API linalg_long matrix_rank(
    const float* A, linalg_long rows, linalg_long columns, linalg_long lda, float tol = -1.F);

LINALG_API linalg_long matrix_rank(
    const double* A, linalg_long rows, linalg_long columns, linalg_long lda, double tol = -1.);

// 2-norm condition number: largest singular value / smallest singular value.
// Returns +infinity when the smallest singular value underflows to exactly
// zero (a numerically singular / rank-deficient matrix).
LINALG_API float matrix_condition_number(
    const float* A, linalg_long rows, linalg_long columns, linalg_long lda);

LINALG_API double matrix_condition_number(
    const double* A, linalg_long rows, linalg_long columns, linalg_long lda);

}  // namespace linalg
