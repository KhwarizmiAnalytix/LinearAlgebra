#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#include "include/matrix_operation/eigen.h"

#include "include/common/linalg_checks.h"

#include <algorithm>
#include <complex>   // needed by OpenBLAS lapack.h on MSVC (no _Complex support)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4190) // C-linkage returning std::complex<> (OpenBLAS lapacke.h)
#endif
#include <lapacke.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "include/common/workspace_arena.h"

#include "blas_runtime.h"

#include "include/common/linalg_profiler.h"

// BLAS/LAPACKE specializations for eigen.h (the native backend lives in
// eigen.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
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
template <typename T> size_t symmetric_eigen_blas_workspace_size(linalg_int /*n*/)
{
    return 0;
}

template <typename T>
size_t general_eigen_blas_workspace_size(linalg_int n, bool compute_eigenvectors)
{
        return 0;
}
}  // namespace

template <>
bool symmetric_eigen_backend<float, linalg_backend::BLAS_LAPACKE>::apply(float* eigenvectors,
    linalg_int                                                                    n,
    linalg_int                                                                    lda,
    float*                                                                        eigenvalues,
    linalg_int                                                                    ldv,
    void*                                                                         workspace,
    size_t                                                                        workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("eigen/blas_lapacke<float>");
    detail::check_symmetric_eigen_inputs(eigenvectors, eigenvalues, n, lda, ldv);
    LINALG_CHECK(workspace_size >= symmetric_eigen_blas_workspace_size<float>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    const auto nn   = static_cast<lapack_int>(n);
    // ssyevd overwrites the input in-place; lda == ldv is required by the caller contract.
    auto       info = LINALG_BLAS_FUNCTION(LAPACKE_ssyevd)(
        LAPACK_ROW_MAJOR, 'V', 'L', nn, eigenvectors, static_cast<lapack_int>(lda), eigenvalues);
    return info == 0;
}

template <>
bool symmetric_eigen_backend<double, linalg_backend::BLAS_LAPACKE>::apply(double* eigenvectors,
    linalg_int                                                                     n,
    linalg_int                                                                     lda,
    double*                                                                        eigenvalues,
    linalg_int ldv,
    void*      workspace,
    size_t     workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("eigen/blas_lapacke<double>");
    detail::check_symmetric_eigen_inputs(eigenvectors, eigenvalues, n, lda, ldv);
    LINALG_CHECK(workspace_size >= symmetric_eigen_blas_workspace_size<double>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    const auto nn   = static_cast<lapack_int>(n);
    // dsyevd overwrites the input in-place; lda == ldv is required by the caller contract.
    auto       info = LINALG_BLAS_FUNCTION(LAPACKE_dsyevd)(
        LAPACK_ROW_MAJOR, 'V', 'L', nn, eigenvectors, static_cast<lapack_int>(lda), eigenvalues);
    return info == 0;
}

template <>
bool general_eigen_backend<float, linalg_backend::BLAS_LAPACKE>::apply(float* eigenvectors,
    linalg_int                                                                  n,
    linalg_int                                                                  lda,
    float*     eigenvalues_real,
    float*     eigenvalues_imag,
    bool       compute_eigenvectors,
    linalg_int ldv,
    void*      workspace,
    size_t     workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("general_eigen/blas_lapacke<float>");
    detail::check_general_eigen_inputs(
        eigenvectors, eigenvalues_real, eigenvalues_imag, compute_eigenvectors, n, lda, ldv);
    LINALG_CHECK(
        workspace_size >= general_eigen_blas_workspace_size<float>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    const auto nn = static_cast<lapack_int>(n);
    if (!compute_eigenvectors)
    {
        auto info = LINALG_BLAS_FUNCTION(LAPACKE_sgeev)(LAPACK_ROW_MAJOR,
            'N',
            'N',
            nn,
            eigenvectors,
            static_cast<lapack_int>(lda),
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            nn,
            nullptr,
            nn);
        return info == 0;
    }
    // sgeev cannot alias the input matrix with the right-eigenvector output; compute
    // into scratch and copy the result back into eigenvectors.
    lapack_int info = LINALG_BLAS_FUNCTION(LAPACKE_sgeev)(LAPACK_ROW_MAJOR,
        'N',
        'V',
        nn,
        eigenvectors,
        static_cast<lapack_int>(lda),
        eigenvalues_real,
        eigenvalues_imag,
        nullptr,
        nn,
        eigenvectors,
        nn);
    return info == 0;
}

template <>
bool general_eigen_backend<double, linalg_backend::BLAS_LAPACKE>::apply(double* eigenvectors,
    linalg_int                                                                    n,
    linalg_int                                                                    lda,
    double*    eigenvalues_real,
    double*    eigenvalues_imag,
    bool       compute_eigenvectors,
    linalg_int ldv,
    void*      workspace,
    size_t     workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("general_eigen/blas_lapacke<double>");
    detail::check_general_eigen_inputs(
        eigenvectors, eigenvalues_real, eigenvalues_imag, compute_eigenvectors, n, lda, ldv);
    LINALG_CHECK(
        workspace_size >= general_eigen_blas_workspace_size<double>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    const auto nn = static_cast<lapack_int>(n);
    if (!compute_eigenvectors)
    {
        auto info = LINALG_BLAS_FUNCTION(LAPACKE_dgeev)(LAPACK_ROW_MAJOR,
            'N',
            'N',
            nn,
            eigenvectors,
            static_cast<lapack_int>(lda),
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            nn,
            nullptr,
            nn);
        return info == 0;
    }
    // dgeev cannot alias the input matrix with the right-eigenvector output; compute
    // into scratch and copy the result back into eigenvectors.
    lapack_int info = LINALG_BLAS_FUNCTION(LAPACKE_dgeev)(LAPACK_ROW_MAJOR,
        'N',
        'V',
        nn,
        eigenvectors,
        static_cast<lapack_int>(lda),
        eigenvalues_real,
        eigenvalues_imag,
        nullptr,
        nn,
        eigenvectors,
        nn);
    return info == 0;
}

template <>
size_t symmetric_eigen_backend<float, linalg_backend::BLAS_LAPACKE>::workspace_size(linalg_int n)
{
    return symmetric_eigen_blas_workspace_size<float>(n);
}

template <>
size_t symmetric_eigen_backend<double, linalg_backend::BLAS_LAPACKE>::workspace_size(linalg_int n)
{
    return symmetric_eigen_blas_workspace_size<double>(n);
}

template <>
size_t general_eigen_backend<float, linalg_backend::BLAS_LAPACKE>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_eigen_blas_workspace_size<float>(n, compute_eigenvectors);
}

template <>
size_t general_eigen_backend<double, linalg_backend::BLAS_LAPACKE>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_eigen_blas_workspace_size<double>(n, compute_eigenvectors);
}

}  // namespace linalg
#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
