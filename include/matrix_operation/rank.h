#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Computes the numerical rank of a matrix.
 */
template <typename T, linalg_backend Backend> struct rank
{
    /**
     * @brief Computes numerical rank by counting singular values above a cutoff.
     *
     * Provide at least workspace_size(rows, columns) bytes of temporary storage.
     *
     * @param[in] A Row-major rows x columns input matrix; left unchanged.
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @param[in] lda Row stride of A in elements; must be at least columns.
     * @param[in] tol Singular-value cutoff. A negative value selects machine epsilon times
     *     max(rows, columns) times the largest singular value.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     * @return Number of singular values strictly greater than the selected cutoff.
     */
    LINALG_API static linalg_long apply(const T* A,
        linalg_long                              rows,
        linalg_long                              columns,
        linalg_long                              lda,
        T                                        tol,
        void*                                    workspace,
        size_t                                   workspace_size,
        void*                                    stream = nullptr);

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(linalg_long rows, linalg_long columns);
};

/**
 * GPU specialization: all matrix, result, pivot, info and workspace pointers
 * refer to device memory. Enqueues work on stream without host transfers or
 * stream synchronization. The caller keeps buffers alive until work completes.
 * CUDA/cuBLAS/cuSOLVER API errors throw; numerical status stays on device.
 */
template <typename T> struct rank<T, linalg_backend::cuda_blas>
{
    LINALG_API static void   apply(const T* A,
        linalg_long                       rows,
        linalg_long                       columns,
        linalg_long                       lda,
        T                                 tol,
        void*                             workspace,
        size_t                            workspace_size,
        linalg_long*                      result,
        void*                             stream = nullptr);
    LINALG_API static size_t workspace_size(linalg_long rows, linalg_long columns);
};

}  // namespace linalg
