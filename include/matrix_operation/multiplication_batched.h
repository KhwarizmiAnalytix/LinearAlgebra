#pragma once

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Batched multiplication of @p count independent square matrices.
 *
 * Computes C_i = A_i * B_i for i in [0, count) using a CUDA kernel. All
 * inputs and outputs are CUDA device pointers; only available when the
 * library is built with LINALG_HAS_CUBLAS.
 *
 * @param[in]  dim    Side length of each square matrix; must be positive.
 * @param[in]  count  Number of independent matrix pairs; must be positive.
 * @param[in]  a      Device pointer to @p count contiguous dim×dim row-major
 *                    matrices (matrix i starts at a + i*dim*dim).
 * @param[in]  b      Device pointer to @p count contiguous dim×dim row-major
 *                    matrices.
 * @param[out] c      Device pointer to @p count contiguous dim×dim row-major
 *                    result matrices.
 * @param[in]  stream Opaque pointer to the CUDA stream (default: nullptr for
 *                    the legacy default stream). The caller is responsible for
 *                    synchronizing before reading @p c on the host.
 */
LINALG_API void batched_matrix_multiplication(linalg_int dim,
    linalg_int                                            count,
    const float*                                          a,
    const float*                                          b,
    float*                                                c,
    void*                                                 stream = nullptr);

LINALG_API void batched_matrix_multiplication(linalg_int dim,
    linalg_int                                            count,
    const double*                                         a,
    const double*                                         b,
    double*                                               c,
    void*                                                 stream = nullptr);

}  // namespace linalg
