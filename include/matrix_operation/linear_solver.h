#pragma once

#include <cstddef>
#include <cstdint>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
/**
 * @brief Factorization and factor-reuse mode for square matrix operations.
 */
enum class linear_solver_type : std::uint8_t
{
    LU_LINEAR_SOLVER               = 1,  ///< Factor the matrix using LU before the operation.
    LU_UPFRONT_LINEAR_SOLVER       = 2,  ///< Reuse existing LU factors and pivots.
    CHOLESKY_LINEAR_SOLVER         = 3,  ///< Factor the matrix using lower Cholesky before the operation.
    CHOLESKY_UPFRONT_LINEAR_SOLVER = 4,  ///< Reuse an existing lower Cholesky factor.
};

/**
 * @brief Solves square linear systems using LU or Cholesky factors.
 */
template <typename T, linalg_backend Backend> struct linear_solver
{
    /**
     * @brief Solves a linear system, replacing its right-hand side with the solution.
     *
     * Cholesky modes require a symmetric positive-definite matrix and use lower triangular
     * factors. Provide at least workspace_size(lda, type) bytes of temporary storage.
     *
     * @param[in,out] m Row-major lda x lda matrix, or existing factors for an UPFRONT mode.
     *     Factorization overwrites the matrix.
     * @param[in,out] pivot Array of lda pivot entries for LU modes; produced during
     *     factorization or supplied with existing factors. Unused for Cholesky.
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[in,out] x Array of lda right-hand-side values; overwritten with the solution.
     * @param[in] type LU or Cholesky solve mode. UPFRONT modes reuse existing factors; other
     *     modes factor the matrix first.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(T* m,
        linalg_int*                 pivot,
        linalg_int                  lda,
        T*                          x,
        linear_solver_type          type,
        void*                       workspace,
        size_t                      workspace_size);

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[in] type LU or Cholesky solve mode. UPFRONT modes reuse existing factors; other
     *     modes factor the matrix first.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(linalg_int lda, linear_solver_type type);
};

}  // namespace linalg
