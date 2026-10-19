#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Computes the economy singular value decomposition A = U * diag(S) * VT.
 */
template <typename T, linalg_backend Backend> struct svd_decomposition
{
    /**
     * @brief Computes the economy factorization A = U * diag(S) * VT.
     *
     * Provide at least workspace_size(rows, columns) bytes of temporary storage.
     *
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @param[in] A Contiguous row-major input matrix of size rows x columns; left unchanged.
     * @param[in] lda Row stride of A in elements; use columns.
     * @param[out] S Array of k nonnegative singular values, where k = min(rows, columns).
     *     Ordering is not guaranteed.
     * @param[out] U Contiguous row-major rows x k matrix of left singular vectors.
     * @param[in] ldu Row stride of U in elements; use k.
     * @param[out] VT Contiguous row-major k x columns transpose of the right singular-vector
     *     matrix.
     * @param[in] ldv Row stride of VT in elements; use columns.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(linalg_long rows,
        linalg_long                          columns,
        T*                                   A,
        linalg_long                          lda,
        T*                                   S,
        T*                                   U,
        linalg_long                          ldu,
        T*                                   VT,
        linalg_long                          ldv,
        void*                                workspace,
        size_t                               workspace_size,
        void*                                stream = nullptr);

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
