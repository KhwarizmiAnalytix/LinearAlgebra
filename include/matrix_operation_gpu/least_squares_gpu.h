#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

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
// `workspace` must point to at least
// `least_squares_solve_workspace_size(type_tag, rows, columns)` bytes of
// caller-owned device memory (the library never allocates its own GPU
// memory; this includes the workspace the internal pseudo_inverse call
// needs); keep it alive and unmodified until work queued on `stream` has
// completed.
LINALG_API size_t least_squares_solve_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t least_squares_solve_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

LINALG_API void least_squares_solve(linalg_long rows,
    linalg_long                                 columns,
    linalg_long                                 nrhs,
    const float*                                  A,
    linalg_long                                 lda,
    const float*                                  B,
    linalg_long                                 ldb,
    float*                                        X,
    linalg_long                                 ldx,
    void*                                          workspace,
    size_t                                         workspace_size,
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
    void*                                          workspace,
    size_t                                         workspace_size,
    cudaStream_t                                  stream = nullptr);

}  // namespace gpu
}  // namespace linalg
