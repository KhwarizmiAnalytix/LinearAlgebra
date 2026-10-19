#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Triangle to compute in a Cholesky factorization.
 */
enum class cholesky_decomposition_enum : char
{
    LOWER_TRIANGULAR = 'L',  ///< Compute the lower factor L in A = L * L^T.
    UPPER_TRIANGULAR = 'U'  ///< Compute the upper factor U in A = U^T * U.
};

/**
 * @brief Cholesky factorization of a symmetric positive-definite matrix.
 */
template <typename T, linalg_backend Backend> struct cholesky_decomposition
{
    /**
     * @brief Factors a symmetric positive-definite matrix in place.
     *
     * @param[in,out] C Row-major lda x lda matrix on input; selected triangular factor on
     *     output.
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[in] type LOWER_TRIANGULAR produces L with C = L * L^T; UPPER_TRIANGULAR produces U
     *     with C = U^T * U.
     * @return True on success; false if factorization fails.
     */
    LINALG_API static bool apply(
        T* C, linalg_int lda, cholesky_decomposition_enum type, void* stream = nullptr);

    /**
     * @brief Propagates derivatives through a Cholesky factorization.
     *
     * @param[in,out] L_aad Row-major lda x lda derivatives of the triangular factor; modified
     *     during propagation.
     * @param[in] L Row-major lda x lda triangular factor from the forward factorization.
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[in] type Triangle used by the forward factorization.
     * @param[in,out] A_aad Row-major lda x lda input-matrix derivatives; contributions are
     *     added to existing values.
     * @return True for a supported triangular factorization.
     */
    LINALG_API static bool apply_aad(
        T* L_aad, const T* L, linalg_int lda, cholesky_decomposition_enum type, T* A_aad);
};

/**
 * GPU specialization: all matrix, result, pivot, info and workspace pointers
 * refer to device memory. Enqueues work on stream without host transfers or
 * stream synchronization. The caller keeps buffers alive until work completes.
 * CUDA/cuBLAS/cuSOLVER API errors throw; numerical status stays on device.
 */
template <typename T> struct cholesky_decomposition<T, linalg_backend::cuda_blas>
{
    LINALG_API static void   apply(T* C,
        linalg_int                  lda,
        cholesky_decomposition_enum type,
        void*                       workspace,
        size_t                      workspace_size,
        int*                        info,
        void*                       stream = nullptr);
    LINALG_API static size_t workspace_size(linalg_int lda, cholesky_decomposition_enum type);
};

}  // namespace linalg
