#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

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

namespace detail
{

// Host-memory-only CPU entry points, both routing through
// linalg::detail::svd_backend (see svd_decomposition.h), which is itself
// the backend-dispatched (scalar / blas_lapacke / mkl) primitive — there is
// no separate backend selection for the reduction done here, so (unlike
// svd_backend) only the `native` specialization of each op below is ever
// defined, and callers always name it explicitly rather than
// kActiveBackend. The template still exists, with the same
// explicit-member-specialization shape as cholesky_backend and friends,
// purely so these ops' call sites look like every other op's. A is
// read-only in both. No GPU implementation exists for either op.
//
// Numerical rank: count of singular values > tol. tol < 0 picks
// tol = eps(T) * max(rows, columns) * largest singular value, matching the
// common LAPACK/NumPy rcond convention.
template <typename T, linalg_backend Backend>
struct matrix_rank_backend
{
    LINALG_API static linalg_long apply(const T* A, linalg_long rows, linalg_long columns,
        linalg_long lda, T tol, void* workspace, size_t workspace_size);
};

// 2-norm condition number: largest singular value / smallest singular value.
// Returns +infinity when the smallest singular value underflows to exactly
// zero (a numerically singular / rank-deficient matrix). `workspace` must
// point to at least `matrix_rank_workspace_size(type_tag, rows, columns)`
// bytes (same shape as matrix_rank_backend above).
template <typename T, linalg_backend Backend>
struct matrix_condition_number_backend
{
    LINALG_API static T apply(
        const T* A, linalg_long rows, linalg_long columns, linalg_long lda, void* workspace, size_t workspace_size);
};

}  // namespace detail

}  // namespace linalg
