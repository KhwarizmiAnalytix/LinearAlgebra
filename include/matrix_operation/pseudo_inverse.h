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
LINALG_API void pseudo_inverse(
    linalg_long rows,
    linalg_long columns,
    const float*  A,
    linalg_long lda,
    float*        Ainv,
    linalg_long ldai,
    float         tol = -1.F);

LINALG_API void pseudo_inverse(
    linalg_long rows,
    linalg_long columns,
    const double* A,
    linalg_long lda,
    double*       Ainv,
    linalg_long ldai,
    double        tol = -1.);

}  // namespace linalg
