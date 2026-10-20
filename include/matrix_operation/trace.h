#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Computes the sum of the diagonal entries of a square matrix.
 */
template <typename T, linalg_backend Backend> struct trace
{
    /**
     * @brief Computes the sum of the diagonal entries of A.
     *
     * @param[in] A Row-major n x n input matrix; left unchanged.
     * @param[in] n Matrix order; must be positive.
     * @param[in] lda Row stride of A in elements; must be at least n.
     * @return Sum of A[i * lda + i] for i from zero to n - 1.
     */
    LINALG_API static T apply(const T* A, linalg_int n, linalg_int lda, void* stream = nullptr);
};

/**
 * GPU specialization: all matrix, result, pivot, info and workspace pointers
 * refer to device memory. Enqueues work on stream without host transfers or
 * stream synchronization. The caller keeps buffers alive until work completes.
 * CUDA/cuBLAS/cuSOLVER API errors throw; numerical status stays on device.
 */
template <typename T> struct trace<T, linalg_backend::CUDA_BLAS>
{
    LINALG_API static void apply(
        const T* A, linalg_int n, linalg_int lda, T* result, void* stream = nullptr);
};

}  // namespace linalg
