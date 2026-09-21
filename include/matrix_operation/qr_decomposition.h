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

// Host-memory-only CPU entry point. Economy ("thin") QR factorization of a
// general rows x columns, row-major matrix A: A = Q * R with
// k = min(rows, columns), Q sized rows x k with orthonormal columns
// (ldq >= k), and R sized k x columns, upper-trapezoidal (ldr >= columns;
// entries below the diagonal of R are written as zero). A is read-only. The
// CPU backend (scalar / blas_lapack / mkl) is a compile-time choice fixed by
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS (MKL takes precedence over BLAS
// over the scalar fallback) — there is no runtime dispatch or per-call
// backend override. No GPU implementation exists for this op.
LINALG_API void qr_decomposition(
    linalg_long rows,
    linalg_long columns,
    const float*  A,
    linalg_long lda,
    float*        Q,
    linalg_long ldq,
    float*        R,
    linalg_long ldr);

LINALG_API void qr_decomposition(
    linalg_long rows,
    linalg_long columns,
    const double* A,
    linalg_long lda,
    double*       Q,
    linalg_long ldq,
    double*       R,
    linalg_long ldr);

}  // namespace linalg
