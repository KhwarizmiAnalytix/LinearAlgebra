#pragma once

#include <cstddef>  // for quarisma_long

#include "include/common/cuda_fwd.h"
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
namespace gpu
{

// `A`, `S`, `U`, `VT` are CUDA device pointers; only available when built
// with LINALG_ENABLE_CUBLAS. Economy SVD (jobu = jobvt = 'S', matching the
// reduced/economy shapes the CPU svd_decomposition also produces): with
// k = min(rows, columns), `S` holds k singular values, `U` is rows x k, and
// `VT` is k x columns, all row-major on return, so that
// A = U * diag(S) * VT — the same decomposition the CPU overload computes,
// via cusolverDnXgesvd instead of LAPACKE_?gesvd. `A` is read-only (copied
// to internal device scratch before factoring), matching the CPU MKL/BLAS
// backends' own contract of not destroying the caller's `A`.
//
// RESTRICTION (unlike the CPU overload, which accepts arbitrary leading
// dimensions): `lda` must equal `columns`, `ldu` must equal
// min(rows, columns), and `ldv` must equal `columns` — i.e. every buffer
// must be tightly packed row-major with no padding. A call with any other
// leading dimension throws rather than silently producing a wrong result.
//
// `info`, a device int* the caller owns, receives cuSOLVER's devInfo (0 =
// success, i < 0 = the i-th argument was invalid, i > 0 = the algorithm did
// not converge). This call never copies `A`/`S`/`U`/`VT` to or from host
// memory, so it never blocks on a device synchronization. `stream`
// (default: the legacy default stream) is the CUDA stream every underlying
// cuSOLVER call and device-to-device copy is issued on.
LINALG_API void svd_decomposition(quarisma_long rows,
    quarisma_long                               columns,
    const float*                                A,
    quarisma_long                               lda,
    float*                                      S,
    float*                                      U,
    quarisma_long                               ldu,
    float*                                      VT,
    quarisma_long                               ldv,
    int*                                        info,
    cudaStream_t                                stream = nullptr);

LINALG_API void svd_decomposition(quarisma_long rows,
    quarisma_long                               columns,
    const double*                               A,
    quarisma_long                               lda,
    double*                                     S,
    double*                                     U,
    quarisma_long                               ldu,
    double*                                     VT,
    quarisma_long                               ldv,
    int*                                        info,
    cudaStream_t                                stream = nullptr);

}  // namespace gpu
}  // namespace linalg
