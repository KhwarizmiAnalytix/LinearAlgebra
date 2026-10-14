#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// Host-memory-only CPU entry points, both routing through
// linalg::svd_decomposition (see svd_decomposition.h), which is itself the
// backend-dispatched (scalar / blas_lapack / mkl) primitive — there is no
// separate backend selection here. A is read-only in both. No GPU
// implementation exists for either op.

// Both `matrix_rank` and `matrix_condition_number` need the same scratch
// shape (they both just reduce over svd_decomposition's singular values
// differently), so they share this one workspace-size query. `workspace`
// must point to at least `matrix_rank_workspace_size(type_tag, rows,
// columns)` bytes of caller-owned host memory (the library never allocates
// its own memory; this includes the workspace the internal
// svd_decomposition call needs — see include/common/workspace_arena.h).
LINALG_API size_t matrix_rank_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t matrix_rank_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

// Numerical rank: count of singular values > tol. tol < 0 picks
// tol = eps(T) * max(rows, columns) * largest singular value, matching the
// common LAPACK/NumPy rcond convention.
LINALG_API linalg_long matrix_rank(const float* A, linalg_long rows, linalg_long columns,
    linalg_long lda, float tol, void* workspace, size_t workspace_size);

LINALG_API linalg_long matrix_rank(const double* A, linalg_long rows, linalg_long columns,
    linalg_long lda, double tol, void* workspace, size_t workspace_size);

// 2-norm condition number: largest singular value / smallest singular value.
// Returns +infinity when the smallest singular value underflows to exactly
// zero (a numerically singular / rank-deficient matrix). `workspace` must
// point to at least `matrix_rank_workspace_size(type_tag, rows, columns)`
// bytes (same shape as matrix_rank above).
LINALG_API float matrix_condition_number(const float* A, linalg_long rows, linalg_long columns,
    linalg_long lda, void* workspace, size_t workspace_size);

LINALG_API double matrix_condition_number(const double* A, linalg_long rows, linalg_long columns,
    linalg_long lda, void* workspace, size_t workspace_size);

}  // namespace linalg
