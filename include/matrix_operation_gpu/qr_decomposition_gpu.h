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

// `A`, `Q`, `R` are CUDA device pointers; only available when built with
// LINALG_ENABLE_CUBLAS. Economy Householder QR (matches the CPU
// qr_decomposition contract exactly): with k = min(rows, columns), `Q` is
// rows x k with orthonormal columns and `R` is k x columns, upper
// trapezoidal (strictly-below-diagonal entries are written as zero), both
// row-major on return, via cusolverDnXgeqrf + cusolverDnXorgqr instead of
// LAPACKE_?geqrf/?orgqr. `A` is read-only (copied to internal device
// scratch before factoring).
//
// RESTRICTION (matching svd_decomposition_gpu's own restriction, for the
// same reason): `lda` must equal `columns`, `ldq` must equal
// min(rows, columns), and `ldr` must equal `columns` — every buffer must be
// tightly packed row-major with no padding.
//
// `info`, a device int* the caller owns, receives cuSOLVER's devInfo (0 =
// success, i < 0 = the i-th argument was invalid). This call never copies
// `A`/`Q`/`R` to or from host memory, so it never blocks on a device
// synchronization. `stream` (default: the legacy default stream) is the
// CUDA stream every underlying cuSOLVER call and device-to-device copy is
// issued on.
LINALG_API void qr_decomposition(linalg_long rows,
    linalg_long                              columns,
    const float*                               A,
    linalg_long                              lda,
    float*                                     Q,
    linalg_long                              ldq,
    float*                                     R,
    linalg_long                              ldr,
    int*                                       info,
    cudaStream_t                               stream = nullptr);

LINALG_API void qr_decomposition(linalg_long rows,
    linalg_long                              columns,
    const double*                              A,
    linalg_long                              lda,
    double*                                    Q,
    linalg_long                              ldq,
    double*                                    R,
    linalg_long                              ldr,
    int*                                       info,
    cudaStream_t                               stream = nullptr);

}  // namespace gpu
}  // namespace linalg
