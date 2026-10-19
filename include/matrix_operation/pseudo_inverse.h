#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Computes the Moore-Penrose pseudo-inverse of a matrix.
 */
template <typename T, linalg_backend Backend> struct pseudo_inverse
{
    /**
     * @brief Computes the Moore-Penrose pseudo-inverse of A.
     *
     * Singular values at or below the cutoff are discarded. Provide at least
     * workspace_size(rows, columns) bytes of temporary storage.
     *
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @param[in] A Row-major rows x columns input matrix; left unchanged.
     * @param[in] lda Row stride of A in elements; must be at least columns.
     * @param[out] Ainv Row-major columns x rows pseudo-inverse matrix.
     * @param[in] ldai Row stride of Ainv in elements; must be at least rows.
     * @param[in] tol Singular-value cutoff. A negative value selects machine epsilon times
     *     max(rows, columns) times the largest singular value.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(linalg_long rows,
        linalg_long                          columns,
        const T*                             A,
        linalg_long                          lda,
        T*                                   Ainv,
        linalg_long                          ldai,
        T                                    tol,
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
