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

// Host-memory-only CPU entry point. Moore-Penrose pseudo-inverse of a
// general rows x columns, row-major matrix A, computed from its SVD
// (A+ = V * S+ * U^T, dropping singular values <= tol). Ainv is sized
// columns x rows (ldai >= rows). A is read-only.
//
// tol < 0 (the default) picks tol = eps(T) * max(rows, columns) * largest
// singular value, matching the common LAPACK/NumPy rcond convention.
//
// This always routes through linalg::svd_decomposition (see
// svd_decomposition.h), which is itself the backend-dispatched (scalar /
// blas_lapack / mkl) primitive — there is no separate backend selection
// here. No GPU implementation exists for this op.
LINALG_API void pseudo_inverse(
    linalg_long rows,
    linalg_long columns,
    const float*  A,
    linalg_long lda,
    float*        Ainv,
    linalg_long ldai,
    float         tol = -1.F);

LINALG_API void pseudo_inverse(
    linalg_long rows,
    linalg_long columns,
    const double* A,
    linalg_long lda,
    double*       Ainv,
    linalg_long ldai,
    double        tol = -1.);

}  // namespace linalg
