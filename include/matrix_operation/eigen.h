#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Computes eigenvalues and eigenvectors of real symmetric matrices.
 */
template <typename T, linalg_backend Backend> struct symmetric_eigen_backend
{
    /**
     * @brief Computes eigenvalues and orthonormal eigenvectors of a real symmetric matrix.
     *
     * On entry eigenvectors holds the n x n input matrix (only the lower triangle is read).
     * On exit it holds the corresponding eigenvectors in its columns. The output satisfies
     * A = V * diag(eigenvalues) * V^T. Provide at least workspace_size(n) bytes of temporary
     * storage.
     *
     * @param[in,out] eigenvectors Row-major n x n matrix. On entry: the symmetric input matrix
     *     (only the lower triangle is read). On exit: eigenvectors as columns.
     * @param[in] n Matrix order; must be positive.
     * @param[in] lda Row stride of eigenvectors on entry in elements; must be at least n.
     * @param[out] eigenvalues Array of n eigenvalues in ascending order.
     * @param[in] ldv Row stride of eigenvectors on exit in elements; must be at least n.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     * @return True on success; false if the iteration does not converge.
     */
    LINALG_API static bool apply(T*         eigenvectors,
        linalg_int                           n,
        linalg_int                           lda,
        T*                                   eigenvalues,
        linalg_int                           ldv,
        void*                                workspace,
        size_t                               workspace_size,
        void*                                stream = nullptr);

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] n Matrix order; must be positive.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(linalg_int n);
};

/**
 * @brief Computes eigenvalues and optional eigenvectors of real square matrices.
 */
template <typename T, linalg_backend Backend> struct general_eigen_backend
{
    /**
     * @brief Computes eigenvalues and optional right eigenvectors of a real square matrix.
     *
     * On entry eigenvectors holds the n x n input matrix; it is overwritten during computation.
     * On exit, when compute_eigenvectors is true, it holds the right eigenvectors as columns.
     * For a conjugate pair at j and j+1 with positive imaginary part at j, columns j and j+1
     * contain the real and imaginary parts of the eigenvector at j. The other eigenvector is
     * its conjugate. Provide workspace_size(n, compute_eigenvectors) bytes of temporary storage.
     *
     * @param[in,out] eigenvectors Row-major n x n matrix. On entry: the input matrix (consumed),
     *     required even when compute_eigenvectors is false. On exit when compute_eigenvectors is
     *     true: right eigenvectors as columns; otherwise its contents are unspecified.
     * @param[in] n Matrix order; must be positive.
     * @param[in] lda Row stride of eigenvectors on entry in elements; must be at least n.
     * @param[out] eigenvalues_real Array of n real parts of the eigenvalues; ordering is not
     *     guaranteed.
     * @param[out] eigenvalues_imag Array of n imaginary parts corresponding to
     *     eigenvalues_real.
     * @param[in] compute_eigenvectors True to compute right eigenvectors; false to compute
     *     eigenvalues only.
     * @param[in] ldv Row stride of eigenvectors on exit in elements; must be at least n when
     *     compute_eigenvectors is true.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     * @return True on success; false if the iteration does not converge.
     *
     * @note Computing eigenvectors for complex eigenvalues may be unsupported and can raise an
     * exception.
     */
    LINALG_API static bool apply(T*         eigenvectors,
        linalg_int                           n,
        linalg_int                           lda,
        T*                                   eigenvalues_real,
        T*                                   eigenvalues_imag,
        bool                                 compute_eigenvectors,
        linalg_int                           ldv,
        void*                                workspace,
        size_t                               workspace_size,
        void*                                stream = nullptr);

    /**
     * @brief Returns the temporary storage required for this operation.
     *
     * @param[in] n Matrix order; must be positive.
     * @param[in] compute_eigenvectors True if eigenvectors will be requested; false for
     *     eigenvalues only.
     * @return Required size of the caller-owned host buffer in bytes; zero if no buffer is
     * needed.
     */
    LINALG_API static size_t workspace_size(linalg_int n, bool compute_eigenvectors);
};

/** CUDA symmetric eigensolver: in-place device matrix and caller-owned device status. */
template <typename T> struct symmetric_eigen_backend<T, linalg_backend::CUDA_BLAS>
{
    LINALG_API static void   apply(T* eigenvectors,
        linalg_int                  n,
        linalg_int                  lda,
        T*                          eigenvalues,
        linalg_int                  ldv,
        void*                       workspace,
        size_t                      workspace_size,
        int*                        info,
        void*                       stream = nullptr);
    LINALG_API static size_t workspace_size(linalg_int n);
};

}  // namespace linalg
