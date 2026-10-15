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
     * The output satisfies A = V * diag(eigenvalues) * V^T. Provide at least workspace_size(n)
     * bytes of temporary storage.
     *
     * @param[in] A Row-major n x n matrix; only the lower triangle is read. Left unchanged.
     * @param[in] n Matrix order; must be positive.
     * @param[in] lda Row stride of A in elements; must be at least n.
     * @param[out] eigenvalues Array of n eigenvalues in ascending order.
     * @param[out] eigenvectors Row-major n x n matrix with corresponding eigenvectors in its
     *     columns.
     * @param[in] ldv Row stride of eigenvectors in elements; must be at least n.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     * @return True on success; false if the iteration does not converge.
     */
    LINALG_API static bool apply(const T* A,
        linalg_int                        n,
        linalg_int                        lda,
        T*                                eigenvalues,
        T*                                eigenvectors,
        linalg_int                        ldv,
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

/**
 * @brief Computes eigenvalues and optional eigenvectors of real square matrices.
 */
template <typename T, linalg_backend Backend> struct general_eigen_backend
{
    /**
     * @brief Computes eigenvalues and optional right eigenvectors of a real square matrix.
     *
     * For a conjugate pair at j and j+1 with positive imaginary part at j, columns j and j+1
     * contain the real and imaginary parts of the eigenvector at j. The other eigenvector is
     * its conjugate. Provide workspace_size(n, eigenvectors != nullptr) bytes of temporary
     * storage.
     *
     * @param[in] A Row-major n x n input matrix; left unchanged.
     * @param[in] n Matrix order; must be positive.
     * @param[in] lda Row stride of A in elements; must be at least n.
     * @param[out] eigenvalues_real Array of n real parts of the eigenvalues; ordering is not
     *     guaranteed.
     * @param[out] eigenvalues_imag Array of n imaginary parts corresponding to
     *     eigenvalues_real.
     * @param[out] eigenvectors Row-major n x n matrix of right eigenvectors, or nullptr to
     *     compute eigenvalues only.
     * @param[in] ldv Row stride of eigenvectors in elements; must be at least n when
     *     eigenvectors is non-null.
     * @param[out] workspace Caller-owned temporary buffer in host memory.
     * @param[in] workspace_size Size of workspace in bytes; must meet the size returned by
     *     workspace_size().
     * @return True on success; false if the iteration does not converge.
     *
     * @note Computing eigenvectors for complex eigenvalues may be unsupported and can raise an
     * exception.
     */
    LINALG_API static bool apply(const T* A,
        linalg_int                        n,
        linalg_int                        lda,
        T*                                eigenvalues_real,
        T*                                eigenvalues_imag,
        T*                                eigenvectors,
        linalg_int                        ldv,
        void*                             workspace,
        size_t                            workspace_size);

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

}  // namespace linalg
