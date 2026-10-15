#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Computes minimum-norm solutions to linear least-squares problems.
 */
template <typename T, linalg_backend Backend> struct least_squares
{
    /**
     * @brief Computes the minimum-norm least-squares solution X to A * X = B.
     *
     * Supports overdetermined, underdetermined, and rank-deficient systems. Provide at least
     * workspace_size(rows, columns) bytes of temporary storage.
     *
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @param[in] nrhs Number of right-hand sides; must be positive.
     * @param[in] A Row-major rows x columns coefficient matrix; left unchanged.
     * @param[in] lda Row stride of A in elements; must be at least columns.
     * @param[in] B Row-major rows x nrhs right-hand-side matrix; left unchanged.
     * @param[in] ldb Row stride of B in elements; must be at least nrhs.
     * @param[out] X Row-major columns x nrhs solution matrix.
     * @param[in] ldx Row stride of X in elements; must be at least nrhs.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(linalg_long rows,
        linalg_long                          columns,
        linalg_long                          nrhs,
        const T*                             A,
        linalg_long                          lda,
        const T*                             B,
        linalg_long                          ldb,
        T*                                   X,
        linalg_long                          ldx,
        void*                                workspace,
        size_t                               workspace_size);

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

}  // namespace linalg
