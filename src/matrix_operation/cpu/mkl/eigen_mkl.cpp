#if LINALG_HAS_MKL

#include "include/matrix_operation/eigen.h"

#include "include/common/linalg_checks.h"

#include <algorithm>
#include <mkl.h>

#include "include/common/workspace_arena.h"

#include "include/common/linalg_profiler.h"

// MKL specializations for eigen.h (the native backend lives in
// eigen.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
// on; the #if below is kept as a second guard.
namespace linalg
{

namespace detail
{

// Same checks as eigen.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_symmetric_eigen_inputs(const void* eigenvectors,
    const void*                                      eigenvalues,
    linalg_int                                       n,
    linalg_int                                       lda,
    linalg_int                                       ldv)
{
    LINALG_CHECK(eigenvalues != nullptr && eigenvectors != nullptr,
        "symmetric_eigen_backend::apply: eigenvalues and eigenvectors must not be null");
    LINALG_CHECK(
        n > 0 && lda > 0 && ldv > 0, "symmetric_eigen_backend::apply: n/lda/ldv must be positive");
}

static void check_general_eigen_inputs(const void* eigenvectors,
    const void*                                    eigenvalues_real,
    const void*                                    eigenvalues_imag,
    bool                                            compute_eigenvectors,
    linalg_int                                      n,
    linalg_int                                      lda,
    linalg_int                                      ldv)
{
    LINALG_CHECK(
        eigenvectors != nullptr && eigenvalues_real != nullptr && eigenvalues_imag != nullptr,
        "general_eigen_backend::apply: eigenvectors, eigenvalues_real, and eigenvalues_imag "
        "must not be null");
    LINALG_CHECK(n > 0 && lda > 0 && !(compute_eigenvectors && ldv <= 0),
        "general_eigen_backend::apply: n/lda/ldv must be positive");
}

}  // namespace detail

namespace
{
template <typename T> size_t symmetric_eigen_mkl_workspace_size(linalg_int /*n*/)
{
    return 0;
}

template <typename T>
size_t general_eigen_mkl_workspace_size(linalg_int n, bool compute_eigenvectors)
{

        return 0;
}
}  // namespace

template <>
bool symmetric_eigen_backend<float, linalg_backend::mkl>::apply(float* eigenvectors,
    linalg_int                                                            n,
    linalg_int                                                            /*lda*/,
    float*                                                                eigenvalues,
    linalg_int                                                            ldv,
    void*                                                                 workspace,
    size_t                                                                workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("eigen/mkl<float>");
    detail::check_symmetric_eigen_inputs(eigenvectors, eigenvalues, n, ldv, ldv);
    LINALG_CHECK(workspace_size >= symmetric_eigen_mkl_workspace_size<float>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    auto info = LAPACKE_ssyevd(LAPACK_ROW_MAJOR, 'V', 'L', n, eigenvectors, ldv, eigenvalues);  // NOLINT
    return info == 0;
}

template <>
bool symmetric_eigen_backend<double, linalg_backend::mkl>::apply(double* eigenvectors,
    linalg_int                                                              n,
    linalg_int                                                              /*lda*/,
    double*                                                                 eigenvalues,
    linalg_int                                                              ldv,
    void*                                                                   workspace,
    size_t                                                                  workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("eigen/mkl<double>");
    detail::check_symmetric_eigen_inputs(eigenvectors, eigenvalues, n, ldv, ldv);
    LINALG_CHECK(workspace_size >= symmetric_eigen_mkl_workspace_size<double>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    auto info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', n, eigenvectors, ldv, eigenvalues);  // NOLINT
    return info == 0;
}

template <>
bool general_eigen_backend<float, linalg_backend::mkl>::apply(float* eigenvectors,
    linalg_int                                                          n,
    linalg_int                                                          /*lda*/,
    float*                                                              eigenvalues_real,
    float*                                                              eigenvalues_imag,
    bool                                                                compute_eigenvectors,
    linalg_int                                                          ldv,
    void*                                                                workspace,
    size_t                                                               workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("general_eigen/mkl<float>");
    detail::check_general_eigen_inputs(
        eigenvectors, eigenvalues_real, eigenvalues_imag, compute_eigenvectors, n, ldv, ldv);
    LINALG_CHECK(
        workspace_size >= general_eigen_mkl_workspace_size<float>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    if (!compute_eigenvectors)
    {
        linalg_int info = LAPACKE_sgeev(  // NOLINT
            LAPACK_ROW_MAJOR, 'N', 'N', n, eigenvectors, ldv, eigenvalues_real, eigenvalues_imag,
            nullptr, n, nullptr, n);
        return info == 0;
    }
    // sgeev cannot alias the input matrix with the right-eigenvector output; compute
    // into scratch and copy the result back into eigenvectors.
    linalg_int info = LAPACKE_sgeev(  // NOLINT
        LAPACK_ROW_MAJOR,
        'N',
        'V',
        n,
        eigenvectors,
        ldv,
        eigenvalues_real,
        eigenvalues_imag,
        nullptr,
        n,
        eigenvectors,
        n);
    return info == 0;
}

template <>
bool general_eigen_backend<double, linalg_backend::mkl>::apply(double* eigenvectors,
    linalg_int                                                            n,
    linalg_int                                                            /*lda*/,
    double*                                                               eigenvalues_real,
    double*                                                               eigenvalues_imag,
    bool                                                                  compute_eigenvectors,
    linalg_int                                                            ldv,
    void*                                                                 workspace,
    size_t                                                                workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("general_eigen/mkl<double>");
    detail::check_general_eigen_inputs(
        eigenvectors, eigenvalues_real, eigenvalues_imag, compute_eigenvectors, n, ldv, ldv);
    LINALG_CHECK(
        workspace_size >= general_eigen_mkl_workspace_size<double>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    if (!compute_eigenvectors)
    {
        linalg_int info = LAPACKE_dgeev(  // NOLINT
            LAPACK_ROW_MAJOR, 'N', 'N', n, eigenvectors, ldv, eigenvalues_real, eigenvalues_imag,
            nullptr, n, nullptr, n);
        return info == 0;
    }
    // dgeev cannot alias the input matrix with the right-eigenvector output; compute
    // into scratch and copy the result back into eigenvectors.
    linalg_int info = LAPACKE_dgeev(  // NOLINT
        LAPACK_ROW_MAJOR,
        'N',
        'V',
        n,
        eigenvectors,
        ldv,
        eigenvalues_real,
        eigenvalues_imag,
        nullptr,
        n,
        eigenvectors,
        n);

    return info == 0;
}

template <> size_t symmetric_eigen_backend<float, linalg_backend::mkl>::workspace_size(linalg_int n)
{
    return symmetric_eigen_mkl_workspace_size<float>(n);
}

template <>
size_t symmetric_eigen_backend<double, linalg_backend::mkl>::workspace_size(linalg_int n)
{
    return symmetric_eigen_mkl_workspace_size<double>(n);
}

template <>
size_t general_eigen_backend<float, linalg_backend::mkl>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_eigen_mkl_workspace_size<float>(n, compute_eigenvectors);
}

template <>
size_t general_eigen_backend<double, linalg_backend::mkl>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_eigen_mkl_workspace_size<double>(n, compute_eigenvectors);
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
