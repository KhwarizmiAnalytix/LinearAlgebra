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
 *
 * One declaration serves every backend (NATIVE, BLAS_LAPACKE, MKL, CUDA_BLAS); no backend
 * specializes the struct itself, so call sites do not vary by backend. Host backends need no
 * scratch space and report status synchronously, so they ignore workspace/workspace_size and
 * always return a definitive result; CUDA_BLAS needs caller-owned device scratch and cannot
 * report the numerical result synchronously, so it fills those parameters and returns only
 * whether the work was launched without an API error (see apply()'s return-value note). This
 * mirrors lu_decomposition's shared workspace/workspace_size parameters
 * (include/matrix_operation/lu_decomposition.h); PyTorch's torch.linalg.cholesky_ex and
 * Eigen's LLT similarly separate a decomposition's own success/failure status from exceptions
 * raised for misuse.
 */
template <typename T, linalg_backend Backend> struct cholesky_decomposition
{
    /**
     * @brief Factors a symmetric positive-definite matrix in place.
     *
     * @param[in,out] C Row-major lda x lda matrix on input; selected triangular factor on
     *     output. Device memory for CUDA_BLAS; host memory otherwise.
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[in] type LOWER_TRIANGULAR produces L with C = L * L^T; UPPER_TRIANGULAR produces U
     *     with C = U^T * U.
     * @param[out] workspace Caller-owned temporary buffer. Device memory for CUDA_BLAS, sized
     *     by workspace_size(); unused (may be null) for host backends.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size() for CUDA_BLAS, ignored for host backends.
     * @param[out] info Optional LAPACK-style status, zero on success and the 1-based index of
     *     the first non-positive-definite leading minor otherwise; may be null. Host memory for
     *     host backends, written before apply() returns. Device memory for CUDA_BLAS, written
     *     asynchronously on stream and only meaningful once the caller has synchronized it.
     * @param[in] stream CUDA_BLAS: stream the work is enqueued on, without host transfers or an
     *     implicit synchronization; the caller keeps C/workspace/info alive until it
     *     synchronizes. Ignored for host backends.
     * @return True on success; false if factorization fails. For CUDA_BLAS this reflects only
     *     that the work was launched without a CUDA/cuBLAS/cuSOLVER API error (such errors
     *     throw instead) -- the numerical result lives in info, not in this return value.
     */
    LINALG_API static bool apply(T* C,
        linalg_int                  lda,
        cholesky_decomposition_enum type,
        void*                       workspace,
        size_t                      workspace_size,
        int*                        info,
        void*                       stream = nullptr);

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

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] lda Matrix order and row stride in elements; the matrix is lda x lda.
     * @param[in] type Triangle to compute.
     * @return Required size of the caller-owned buffer in bytes; always zero for host backends.
     */
    LINALG_API static size_t workspace_size(linalg_int lda, cholesky_decomposition_enum type);
};

}  // namespace linalg
