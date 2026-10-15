#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Computes the economy QR factorization A = Q * R.
 */
template <typename T, linalg_backend Backend> struct qr_decomposition
{
    /**
     * @brief Computes the economy factorization A = Q * R.
     *
     * Provide at least workspace_size(rows, columns) bytes of temporary storage.
     *
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @param[in] A Row-major input matrix of size rows x columns; left unchanged.
     * @param[in] lda Row stride of A in elements; must be at least columns.
     * @param[out] Q Row-major rows x k matrix with orthonormal columns, where k = min(rows,
     *     columns).
     * @param[in] ldq Row stride of Q in elements; must be at least k.
     * @param[out] R Row-major k x columns upper trapezoidal matrix; entries below the diagonal
     *     are zero.
     * @param[in] ldr Row stride of R in elements; must be at least columns.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(linalg_long rows,
        linalg_long                          columns,
        const T*                             A,
        linalg_long                          lda,
        T*                                   Q,
        linalg_long                          ldq,
        T*                                   R,
        linalg_long                          ldr,
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
