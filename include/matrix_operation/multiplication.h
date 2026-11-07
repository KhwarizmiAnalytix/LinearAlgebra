#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief General matrix-matrix multiplication with optional transposition (BLAS GEMM).
 */
template <typename T, linalg_backend Backend> struct multiplication
{
    /**
     * @brief Computes c = alpha * op(a) * op(b) + beta * c.
     *
     * Follows the BLAS ?gemm contract on row-major storage: when beta is zero, c is
     * write-only (its prior contents, including NaN or Inf, are not read); when alpha is
     * zero, the product is skipped and c is only scaled by beta. c must not overlap a or b.
     *
     * Provide at least workspace_size(rows, columns, depth) bytes of temporary storage.
     *
     * @param[in] transpose_a True to use the transpose of a; false to use a unchanged.
     * @param[in] transpose_b True to use the transpose of b; false to use b unchanged.
     * @param[in] rows Number of rows in op(a) and c.
     * @param[in] columns Number of columns in op(b) and c.
     * @param[in] depth Number of columns in op(a), equal to the number of rows in op(b).
     * @param[in] alpha Scale applied to the product op(a) * op(b).
     * @param[in] a First row-major matrix; stored as rows x depth, or depth x rows when
     *     transposed.
     * @param[in] lda Stored row stride of a in elements.
     * @param[in] b Second row-major matrix; stored as depth x columns, or columns x depth when
     *     transposed.
     * @param[in] ldb Stored row stride of b in elements.
     * @param[in] beta Scale applied to the prior contents of c.
     * @param[in,out] c Row-major result matrix of size rows x columns; read only when beta is
     *     nonzero.
     * @param[in] ldc Row stride of c in elements.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     * @param[in] stream Backend stream or queue; ignored by CPU backends.
     */
    LINALG_API static void apply(bool transpose_a,
        bool                          transpose_b,
        linalg_int                    rows,
        linalg_int                    columns,
        linalg_int                    depth,
        T                             alpha,
        const T*                      a,
        linalg_int                    lda,
        const T*                      b,
        linalg_int                    ldb,
        T                             beta,
        T*                            c,
        linalg_int                    ldc,
        void*                         workspace,
        size_t                        workspace_size,
        void*                         stream = nullptr);

    /**
     * @brief Computes c = op(a) * op(b), replacing the contents of c.
     *
     * Shorthand for the GEMM overload with alpha = 1 and beta = 0.
     */
    static void apply(bool transpose_a,
        bool               transpose_b,
        linalg_int         rows,
        linalg_int         columns,
        linalg_int         depth,
        const T*           a,
        linalg_int         lda,
        const T*           b,
        linalg_int         ldb,
        T*                 c,
        linalg_int         ldc,
        void*              workspace,
        size_t             workspace_size,
        void*              stream = nullptr)
    {
        apply(transpose_a,
            transpose_b,
            rows,
            columns,
            depth,
            T(1),
            a,
            lda,
            b,
            ldb,
            T(0),
            c,
            ldc,
            workspace,
            workspace_size,
            stream);
    }

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * The requirement does not depend on alpha or beta.
     *
     * @param[in] rows Number of rows in the result matrix.
     * @param[in] columns Number of columns in the result matrix.
     * @param[in] depth Shared inner dimension of the two factors.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(linalg_int rows, linalg_int columns, linalg_int depth);
};

}  // namespace linalg
