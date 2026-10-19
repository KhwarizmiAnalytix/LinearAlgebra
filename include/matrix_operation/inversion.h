#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"

namespace linalg
{

/**
 * @brief Inverts square matrices using LU or Cholesky factors.
 */
template <typename T, linalg_backend Backend> struct inversion
{
    /**
     * @brief Computes the inverse of a square matrix in place.
     *
     * Cholesky modes require a symmetric positive-definite matrix and use lower triangular
     * factors. Provide at least workspace_size(lda, type) bytes of temporary storage.
     *
     * @param[in,out] m Row-major lda x lda matrix or existing factors; overwritten with the
     *     inverse. For Cholesky, use the lower triangle of the result.
     * @param[in,out] pivot Array of lda pivot entries for LU modes; produced during
     *     factorization or supplied with existing factors. Unused for Cholesky.
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[in] type LU or Cholesky solve mode. UPFRONT modes reuse existing factors; other
     *     modes factor the matrix first.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(T* m,
        linalg_int*                 pivot,
        linalg_int                  lda,
        linalg::linear_solver_type  type,
        void*                       workspace,
        size_t                      workspace_size,
        void*                       stream = nullptr);

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[in] type LU or Cholesky solve mode. UPFRONT modes reuse existing factors; other
     *     modes factor the matrix first.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(linalg_int lda, linalg::linear_solver_type type);
};

}  // namespace linalg
