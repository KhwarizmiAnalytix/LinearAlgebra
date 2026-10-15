#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

inline constexpr linalg_backend kMatmulActiveBackend =
#if LINALG_HAS_MKL
    linalg_backend::mkl;
#elif LINALG_HAS_BLAS
    linalg_backend::blas_lapacke;
#else
    linalg_backend::native;
#endif

/**
 * @brief Multiplies two matrices with optional transposition.
 */
template <typename T, linalg_backend Backend> struct multiplication
{
    /**
     * @brief Computes c = op(a) * op(b), replacing the contents of c.
     *
     * Provide at least workspace_size(rows, columns, depth) bytes of temporary storage.
     *
     * @param[in] transpose_a True to use the transpose of a; false to use a unchanged.
     * @param[in] transpose_b True to use the transpose of b; false to use b unchanged.
     * @param[in] rows Number of rows in op(a) and c.
     * @param[in] columns Number of columns in op(b) and c.
     * @param[in] depth Number of columns in op(a), equal to the number of rows in op(b).
     * @param[in] a First row-major matrix; stored as rows x depth, or depth x rows when
     *     transposed.
     * @param[in] lda Stored row stride of a in elements.
     * @param[in] b Second row-major matrix; stored as depth x columns, or columns x depth when
     *     transposed.
     * @param[in] ldb Stored row stride of b in elements.
     * @param[out] c Row-major result matrix of size rows x columns.
     * @param[in] ldc Row stride of c in elements.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(bool transpose_a,
        bool                          transpose_b,
        linalg_int                    rows,
        linalg_int                    columns,
        linalg_int                    depth,
        const T*                      a,
        linalg_int                    lda,
        const T*                      b,
        linalg_int                    ldb,
        T*                            c,
        linalg_int                    ldc,
        void*                         workspace,
        size_t                        workspace_size);

    /**
     * @brief Returns the temporary storage required for this operation.
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
