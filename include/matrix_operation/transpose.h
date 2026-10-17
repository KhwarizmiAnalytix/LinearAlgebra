#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Transposes a matrix in place.
 */
template <typename T, linalg_backend Backend> struct transpose
{
    /**
     * @brief Transposes a contiguous row-major matrix in place.
     *
     * Provide at least workspace_size(rows, columns) bytes of temporary storage.
     *
     * @param[in] rows Number of matrix rows; must be positive.
     * @param[in] columns Number of matrix columns; must be positive.
     * @param[in,out] m Matrix buffer containing rows x columns entries; represents a columns x
     *     rows matrix on return.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     */
    LINALG_API static void apply(
        linalg_long rows, linalg_long columns, T* m, void* workspace, size_t workspace_size);

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
