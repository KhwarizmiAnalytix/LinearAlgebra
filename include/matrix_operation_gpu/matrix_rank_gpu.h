#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
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
namespace gpu
{

// `A` is a CUDA device pointer; only available when built with
// LINALG_ENABLE_CUBLAS. Both route through linalg::gpu::svd_decomposition
// and then reduce over the (small, length k = min(rows, columns))
// singular-value vector on the host — like matrix_determinant, this
// performs one internal device synchronization and copies that small
// vector back (never the full rows x columns matrix). `A` is read-only.
//
// RESTRICTION (matching svd_decomposition_gpu): `lda` must equal `columns`.
//
// Numerical rank: count of singular values > tol. tol < 0 (the default)
// picks tol = eps(T) * max(rows, columns) * largest singular value.
LINALG_API linalg_long matrix_rank(const float* A,
    linalg_long                                rows,
    linalg_long                                columns,
    linalg_long                                lda,
    float                                        tol    = -1.F,
    cudaStream_t                                 stream = nullptr);

LINALG_API linalg_long matrix_rank(const double* A,
    linalg_long                                 rows,
    linalg_long                                 columns,
    linalg_long                                 lda,
    double                                        tol    = -1.,
    cudaStream_t                                  stream = nullptr);

// 2-norm condition number: largest singular value / smallest singular
// value. Returns +infinity when the smallest singular value underflows to
// exactly zero.
LINALG_API float matrix_condition_number(
    const float* A, linalg_long rows, linalg_long columns, linalg_long lda, cudaStream_t stream = nullptr);

LINALG_API double matrix_condition_number(const double* A,
    linalg_long                                       rows,
    linalg_long                                       columns,
    linalg_long                                       lda,
    cudaStream_t                                        stream = nullptr);

}  // namespace gpu
}  // namespace linalg
