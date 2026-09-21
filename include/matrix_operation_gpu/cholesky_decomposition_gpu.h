#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/cholesky_decomposition.h"

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

// `C` is a CUDA device pointer; only available when built with
// LINALG_ENABLE_CUBLAS. Success/failure (cuSOLVER's devInfo: 0 means
// success, i > 0 means the leading i-th minor is not positive definite) is
// written into `info`, itself a device int* the caller owns — this call
// never copies to or from host memory, so it never blocks on a device
// synchronization. Inspect `info` explicitly (e.g. cudaMemcpyAsync on your
// own stream) only when/if you need the result, rather than paying that
// latency on every call. `stream` (default: the legacy default stream) is
// the CUDA stream the underlying cuSOLVER call is issued on.
LINALG_API void cholesky_decomposition(float* C,
    linalg_int                              lda,
    linalg::cholesky_decomposition_enum       type,
    int*                                      info,
    cudaStream_t                              stream = nullptr);

LINALG_API void cholesky_decomposition(double* C,
    linalg_int                               lda,
    linalg::cholesky_decomposition_enum        type,
    int*                                       info,
    cudaStream_t                               stream = nullptr);

}  // namespace gpu
}  // namespace linalg
