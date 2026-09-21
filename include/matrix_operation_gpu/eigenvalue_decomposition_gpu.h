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

// `A`, `eigenvalues`, `eigenvectors` are CUDA device pointers; only
// available when built with LINALG_ENABLE_CUBLAS. Real symmetric
// eigenvalue decomposition via cusolverDnXsyevd, matching the CPU
// symmetric_eigenvalue_decomposition contract exactly: eigenvalues are
// written ascending, and the corresponding orthonormal eigenvectors as the
// columns of `eigenvectors` (n x n, row-major, ldv >= n). A is read-only
// (copied to internal device scratch before factoring); only its lower
// triangle is referenced.
//
// `info`, a device int* the caller owns, receives cuSOLVER's devInfo (0 =
// success, i < 0 = the i-th argument was invalid, i > 0 = the algorithm did
// not converge). This call never copies to or from host memory, so it never
// blocks on a device synchronization. `stream` (default: the legacy default
// stream) is the CUDA stream every underlying cuSOLVER call and
// device-to-device copy is issued on.
//
// UNLIKE the CPU header's pair, there is no general (non-symmetric)
// linalg::gpu::eigenvalue_decomposition: cuSOLVER's dense API
// (cusolverDn) has no geev-equivalent routine for real non-symmetric
// matrices as of this writing — nothing to call into, so nothing is
// declared here (matching this codebase's convention of documenting a
// missing vendor routine rather than faking coverage, e.g.
// svd_decomposition.h's own "no GPU implementation exists for this op"
// note). Use the CPU linalg::eigenvalue_decomposition
// (include/matrix_operation/eigenvalue_decomposition.h) for the general
// case.
LINALG_API void symmetric_eigenvalue_decomposition(const float* A,
    linalg_int                                                n,
    linalg_int                                                lda,
    float*                                                      eigenvalues,
    float*                                                      eigenvectors,
    linalg_int                                                ldv,
    int*                                                         info,
    cudaStream_t                                                stream = nullptr);

LINALG_API void symmetric_eigenvalue_decomposition(const double* A,
    linalg_int                                                 n,
    linalg_int                                                 lda,
    double*                                                      eigenvalues,
    double*                                                      eigenvectors,
    linalg_int                                                 ldv,
    int*                                                          info,
    cudaStream_t                                                 stream = nullptr);

}  // namespace gpu
}  // namespace linalg
