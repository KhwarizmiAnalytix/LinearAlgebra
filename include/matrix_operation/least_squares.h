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

// Host-memory-only CPU entry point. Solves the general linear least-squares
// problem min_X ||A*X - B||_2 for a rows x columns, row-major A and a
// rows x nrhs, row-major B: overdetermined (rows > columns) gives the
// least-squares solution, underdetermined (rows < columns) gives the
// minimum-norm solution, and rank-deficient A is handled the same way via
// the pseudo-inverse. X is sized columns x nrhs (ldx >= nrhs). A and B are
// read-only.
//
// Always routes through linalg::pseudo_inverse (X = A+ * B), which is
// itself backend-dispatched via linalg::svd_decomposition — there is no
// separate backend selection here. No GPU implementation exists for this
// op.
LINALG_API void least_squares_solve(
    linalg_long rows,
    linalg_long columns,
    linalg_long nrhs,
    const float*  A,
    linalg_long lda,
    const float*  B,
    linalg_long ldb,
    float*        X,
    linalg_long ldx);

LINALG_API void least_squares_solve(
    linalg_long rows,
    linalg_long columns,
    linalg_long nrhs,
    const double* A,
    linalg_long lda,
    const double* B,
    linalg_long ldb,
    double*       X,
    linalg_long ldx);

}  // namespace linalg
