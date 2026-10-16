#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Computes an integer matrix power.
 */
template <typename T, linalg_backend Backend> struct power
{
    /**
     * @brief Computes result = A^exponent.
     *
     * exponent == 0 yields the identity matrix (A need not be invertible). exponent < 0
     * inverts A before raising the inverse to |exponent|. Provide at least
     * workspace_size(n) bytes of temporary storage.
     *
     * @param[in] A Row-major n x n input matrix; left unchanged unless result aliases A.
     * @param[in] n Matrix order; must be positive.
     * @param[in] lda Row stride of A in elements; must be at least n.
     * @param[in] exponent Integer power to raise A to; negative values require A invertible.
     * @param[out] result Row-major n x n matrix power; may alias A.
     * @param[in] ldc Row stride of result in elements; must be at least n.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(const T* A,
        linalg_int                        n,
        linalg_int                        lda,
        linalg_int                        exponent,
        T*                                result,
        linalg_int                        ldc,
        void*                             workspace,
        size_t                            workspace_size);

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] n Matrix order; must be positive.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(linalg_int n);
};

}  // namespace linalg
