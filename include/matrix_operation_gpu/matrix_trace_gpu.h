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
// LINALG_HAS_CUBLAS. Sum of the diagonal of a square, row-major n x n
// device matrix. Like matrix_determinant_gpu, this performs one internal
// device synchronization and a single-scalar device-to-host copy — the
// diagonal is summed on-device by a small kernel first, so the full n x n
// matrix is never downloaded.
// `workspace` must point to at least `matrix_trace_workspace_size()` bytes
// of caller-owned device memory (the library never allocates its own GPU
// memory) — a single scalar, but still caller-supplied for consistency with
// every other linalg::gpu::* entry point.
LINALG_API size_t matrix_trace_workspace_size(const float* /*type_tag*/);
LINALG_API size_t matrix_trace_workspace_size(const double* /*type_tag*/);

LINALG_API float matrix_trace(const float* A, linalg_int n, linalg_int lda, void* workspace,
    size_t workspace_size, cudaStream_t stream = nullptr);

LINALG_API double matrix_trace(const double* A, linalg_int n, linalg_int lda, void* workspace,
    size_t workspace_size, cudaStream_t stream = nullptr);

}  // namespace gpu
}  // namespace linalg
