#pragma once

#include <cstddef>  // for linalg_long

#include "include/common/linear_algebra_export.h"  // for LINALG_API

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define linalg_int __int64
#define linalg_long unsigned __int64
#else
#define linalg_int long long int
#define linalg_long unsigned long long int
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
    linalg_long rows,
    linalg_long columns,
    float*      A,
    linalg_long lda,
    float*      S,
    float*      U,
    linalg_long ldu,
    float*      VT,
    linalg_long ldv);

LINALG_API void svd_decomposition(
    linalg_long rows,
    linalg_long columns,
    double*     A,
    linalg_long lda,
    double*     S,
    double*     U,
    linalg_long ldu,
    double*     VT,
    linalg_long ldv);

}  // namespace linalg
