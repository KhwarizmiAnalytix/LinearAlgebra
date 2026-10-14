#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// Host-memory-only CPU entry point. Moore-Penrose pseudo-inverse of a
// general rows x columns, row-major matrix A, computed from its SVD
// (A+ = V * S+ * U^T, dropping singular values <= tol). Ainv is sized
// columns x rows (ldai >= rows). A is read-only.
//
// tol < 0 (the default) picks tol = eps(T) * max(rows, columns) * largest
// singular value, matching the common LAPACK/NumPy rcond convention.
//
// This always routes through linalg::svd_decomposition (see
// svd_decomposition.h), which is itself the backend-dispatched (scalar /
// blas_lapack / mkl) primitive — there is no separate backend selection
// here. For device (CUDA) pointers, see linalg::gpu::pseudo_inverse in
// include/matrix_operation_gpu/pseudo_inverse_gpu.h.
// `workspace` must point to at least `pseudo_inverse_workspace_size(
// type_tag, rows, columns)` bytes of caller-owned host memory (the library
// never allocates its own memory; this includes the workspace the internal
// svd_decomposition call needs — see include/common/workspace_arena.h).
LINALG_API size_t pseudo_inverse_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t pseudo_inverse_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

LINALG_API void pseudo_inverse(
    linalg_long rows,
    linalg_long columns,
    const float*  A,
    linalg_long lda,
    float*        Ainv,
    linalg_long ldai,
    float         tol,
    void*         workspace,
    size_t        workspace_size);

LINALG_API void pseudo_inverse(
    linalg_long rows,
    linalg_long columns,
    const double* A,
    linalg_long lda,
    double*       Ainv,
    linalg_long ldai,
    double        tol,
    void*         workspace,
    size_t        workspace_size);

}  // namespace linalg
