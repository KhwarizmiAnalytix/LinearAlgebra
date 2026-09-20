#pragma once

#include <cstddef>  // for quarisma_long

#include "include/common/linear_algebra_export.h"  // for LINALG_API

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif


namespace linalg
{
// Host-memory-only CPU entry point (no GPU implementation exists for this
// op — the row-major/column-major and U/V-swap bookkeeping cuSOLVER's gesvd
// would need has not been validated against real hardware). The CPU backend
// (scalar / blas_lapack / mkl) is a compile-time choice fixed by
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS (MKL takes precedence over BLAS
// over the scalar fallback) — there is no runtime dispatch or per-call
// backend override.
LINALG_API void svd_decomposition(
    quarisma_long rows,
    quarisma_long columns,
    float*      A,
    quarisma_long lda,
    float*      S,
    float*      U,
    quarisma_long ldu,
    float*      VT,
    quarisma_long ldv);

LINALG_API void svd_decomposition(
    quarisma_long rows,
    quarisma_long columns,
    double*     A,
    quarisma_long lda,
    double*     S,
    double*     U,
    quarisma_long ldu,
    double*     VT,
    quarisma_long ldv);

}  // namespace linalg
