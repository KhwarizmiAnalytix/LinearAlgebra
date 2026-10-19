#pragma once

#include <cstddef>
#include <cstdint>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Matrix norm to compute.
 */
enum class norm_type : std::uint8_t
{
    FROBENIUS     = 1,  ///< Square root of the sum of squared entries.
    ONE           = 2,  ///< Maximum absolute column sum.
    INFINITY_NORM = 3,  ///< Maximum absolute row sum.
    TWO           = 4,  ///< Spectral norm: largest singular value.
};

/**
 * @brief Computes Frobenius, 1-norm, infinity-norm, or spectral matrix norms.
 */
template <typename T, linalg_backend Backend> struct norm
{
    /**
     * @brief Computes the selected norm of a matrix.
     *
     * Provide at least workspace_size(rows, columns, type) bytes of temporary storage. Only TWO
     * requires temporary storage.
     *
     * @param[in] A Row-major rows x columns input matrix; left unchanged.
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @param[in] lda Row stride of A in elements; must be at least columns.
     * @param[in] type Norm to compute: FROBENIUS, ONE, INFINITY_NORM, or TWO.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     * @return The selected matrix norm.
     */
    LINALG_API static T apply(const T* A,
        linalg_long                    rows,
        linalg_long                    columns,
        linalg_long                    lda,
        norm_type                      type,
        void*                          workspace,
        size_t                         workspace_size,
        void*                          stream = nullptr);

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @param[in] type Norm that will be requested.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(
        linalg_long rows, linalg_long columns, norm_type type);
};

}  // namespace linalg
