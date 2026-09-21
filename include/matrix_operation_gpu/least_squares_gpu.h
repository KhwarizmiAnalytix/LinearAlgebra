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

// `A`, `B`, `X` are CUDA device pointers; only available when built with
// LINALG_ENABLE_CUBLAS. Solves min_X ||A*X - B||_2, matching the CPU
// least_squares_solve contract exactly (over/under-determined,
// rank-deficient A handled the same way via the pseudo-inverse): `X`
// (columns x nrhs, row-major, ldx >= nrhs) = linalg::gpu::pseudo_inverse(A)
// * `B` (rows x nrhs, row-major). `A` and `B` are read-only.
//
// RESTRICTION (matching pseudo_inverse_gpu / svd_decomposition_gpu): `lda`
// must equal `columns`, `ldb` must equal `nrhs`, and `ldx` must equal
// `nrhs` — every buffer must be tightly packed row-major with no padding.
LINALG_API void least_squares_solve(linalg_long rows,
    linalg_long                                 columns,
    linalg_long                                 nrhs,
    const float*                                  A,
    linalg_long                                 lda,
    const float*                                  B,
    linalg_long                                 ldb,
    float*                                        X,
    linalg_long                                 ldx,
    cudaStream_t                                  stream = nullptr);

LINALG_API void least_squares_solve(linalg_long rows,
    linalg_long                                 columns,
    linalg_long                                 nrhs,
    const double*                                 A,
    linalg_long                                 lda,
    const double*                                 B,
    linalg_long                                 ldb,
    double*                                       X,
    linalg_long                                 ldx,
    cudaStream_t                                  stream = nullptr);

}  // namespace gpu
}  // namespace linalg
