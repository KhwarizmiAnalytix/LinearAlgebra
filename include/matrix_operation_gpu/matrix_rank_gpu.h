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
// LINALG_HAS_CUBLAS. Both route through linalg::gpu::svd_decomposition
// and then reduce over the (small, length k = min(rows, columns))
// singular-value vector on the host — like matrix_determinant, this
// performs one internal device synchronization and copies that small
// vector back (never the full rows x columns matrix). `A` is read-only.
//
// RESTRICTION (matching svd_decomposition_gpu): `lda` must equal `columns`.
//
// Numerical rank: count of singular values > tol. tol < 0 (the default)
// picks tol = eps(T) * max(rows, columns) * largest singular value.
// Both `matrix_rank` and `matrix_condition_number` need the same scratch
// shape (they both just reduce over svd_decomposition's singular values
// differently), so they share this one workspace-size query. `workspace`
// must point to at least `matrix_rank_workspace_size(type_tag, rows,
// columns)` bytes of caller-owned device memory (the library never
// allocates its own GPU memory; this includes the workspace the internal
// svd_decomposition call needs).
LINALG_API size_t matrix_rank_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t matrix_rank_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

LINALG_API linalg_long matrix_rank(const float* A,
    linalg_long                                rows,
    linalg_long                                columns,
    linalg_long                                lda,
    float                                        tol,
    void*                                        workspace,
    size_t                                       workspace_size,
    cudaStream_t                                 stream = nullptr);

LINALG_API linalg_long matrix_rank(const double* A,
    linalg_long                                 rows,
    linalg_long                                 columns,
    linalg_long                                 lda,
    double                                        tol,
    void*                                         workspace,
    size_t                                        workspace_size,
    cudaStream_t                                  stream = nullptr);

// 2-norm condition number: largest singular value / smallest singular
// value. Returns +infinity when the smallest singular value underflows to
// exactly zero. `workspace` must point to at least
// `matrix_rank_workspace_size(type_tag, rows, columns)` bytes (same shape
// as matrix_rank above).
LINALG_API float matrix_condition_number(const float* A, linalg_long rows, linalg_long columns,
    linalg_long lda, void* workspace, size_t workspace_size, cudaStream_t stream = nullptr);

LINALG_API double matrix_condition_number(const double* A,
    linalg_long                                       rows,
    linalg_long                                       columns,
    linalg_long                                       lda,
    void*                                               workspace,
    size_t                                              workspace_size,
    cudaStream_t                                        stream = nullptr);

}  // namespace gpu
}  // namespace linalg
