#pragma once

#include <cstddef>

#include "include/common/backend.h"
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
// This always routes through linalg::svd_backend (see svd_decomposition.h),
// which is itself the backend-dispatched (scalar / blas_lapacke / mkl)
// primitive — there is no separate backend selection for the reduction done
// here, so only the `native` specialization below is ever defined, and
// callers always name it explicitly rather than kActiveBackend. The template
// still exists, with the same explicit-member-specialization shape as
// cholesky_backend and friends, purely so this op's call site looks like
// every other op's. For device (CUDA) pointers, see linalg::gpu::pseudo_inverse
// in include/matrix_operation_gpu/pseudo_inverse_gpu.h.
template <typename T, linalg_backend Backend> struct pseudo_inverse_backend
{
    LINALG_API static void apply(linalg_long rows,
        linalg_long                          columns,
        const T*                             A,
        linalg_long                          lda,
        T*                                   Ainv,
        linalg_long                          ldai,
        T                                    tol,
        void*                                workspace,
        size_t                               workspace_size);

    LINALG_API static size_t workspace_size(linalg_long rows, linalg_long columns);
};

}  // namespace linalg
