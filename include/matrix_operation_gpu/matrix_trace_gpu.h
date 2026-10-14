#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
namespace gpu
{

// `A` is a CUDA device pointer; only available when built with
// LINALG_ENABLE_CUBLAS. Sum of the diagonal of a square, row-major n x n
// device matrix. Like matrix_determinant_gpu, this performs one internal
// device synchronization and a single-scalar device-to-host copy — the
// diagonal is summed on-device by a small kernel first, so the full n x n
// matrix is never downloaded.
LINALG_API float matrix_trace(
    const float* A, linalg_int n, linalg_int lda, cudaStream_t stream = nullptr);

LINALG_API double matrix_trace(
    const double* A, linalg_int n, linalg_int lda, cudaStream_t stream = nullptr);

}  // namespace gpu
}  // namespace linalg
