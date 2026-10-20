#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief LU factorization of a square matrix.
 */
template <typename T, linalg_backend Backend> struct lu_decomposition
{
    /**
     * @brief Factors a square matrix in place into L and U.
     *
     * Provide at least workspace_size(lda) bytes of temporary storage.
     *
     * @param[in,out] m Row-major lda x lda matrix; overwritten with L below the diagonal and U
     *     on and above it. L has an implicit unit diagonal.
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[out] pivot Array of lda pivot entries to retain for subsequent LU solves or
     *     inversion.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     * @return True on success; false if factorization fails.
     */
    LINALG_API static bool apply(T* m,
        linalg_int                   lda,
        linalg_int*                  pivot,
        void*                        workspace,
        size_t                       workspace_size,
        void*                        stream = nullptr);

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(linalg_int lda);
};

/**
 * GPU specialization: all matrix, result, pivot, info and workspace pointers
 * refer to device memory. Enqueues work on stream without host transfers or
 * stream synchronization. The caller keeps buffers alive until work completes.
 * CUDA/cuBLAS/cuSOLVER API errors throw; numerical status stays on device.
 */
template <typename T> struct lu_decomposition<T, linalg_backend::CUDA_BLAS>
{
    LINALG_API static void   apply(T* m,
        linalg_int                  lda,
        int*                        pivot,
        void*                       workspace,
        size_t                      workspace_size,
        int*                        info,
        void*                       stream = nullptr);
    LINALG_API static size_t workspace_size(linalg_int lda);
};

}  // namespace linalg
