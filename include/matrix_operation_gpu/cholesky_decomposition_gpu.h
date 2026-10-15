#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/common/types.h"

namespace linalg
{
namespace gpu
{

// `C` is a CUDA device pointer; only available when built with
// LINALG_HAS_CUBLAS. Success/failure (cuSOLVER's devInfo: 0 means
// success, i > 0 means the leading i-th minor is not positive definite) is
// written into `info`, itself a device int* the caller owns — this call
// never copies to or from host memory, so it never blocks on a device
// synchronization. Inspect `info` explicitly (e.g. cudaMemcpyAsync on your
// own stream) only when/if you need the result, rather than paying that
// latency on every call. `stream` (default: the legacy default stream) is
// the CUDA stream the underlying cuSOLVER call is issued on.
// `workspace` must point to at least
// `cholesky_decomposition_workspace_size(type_tag, lda, type)` bytes of
// caller-owned device memory (the library never allocates its own GPU
// memory); keep it alive and unmodified until work queued on `stream` has
// completed.
LINALG_API size_t cholesky_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_int lda, linalg::cholesky_decomposition_enum type);
LINALG_API size_t cholesky_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_int lda, linalg::cholesky_decomposition_enum type);

LINALG_API void cholesky_decomposition(float* C,
    linalg_int                              lda,
    linalg::cholesky_decomposition_enum       type,
    int*                                      info,
    void*                                     workspace,
    size_t                                    workspace_size,
    cudaStream_t                              stream = nullptr);

LINALG_API void cholesky_decomposition(double* C,
    linalg_int                               lda,
    linalg::cholesky_decomposition_enum        type,
    int*                                       info,
    void*                                      workspace,
    size_t                                     workspace_size,
    cudaStream_t                               stream = nullptr);

}  // namespace gpu
}  // namespace linalg
