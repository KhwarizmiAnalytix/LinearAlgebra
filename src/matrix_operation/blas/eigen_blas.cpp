#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#include "include/matrix_operation/eigen.h"

#include "include/common/linalg_checks.h"

#include <algorithm>
#include <complex>   // needed by OpenBLAS lapack.h on MSVC (no _Complex support)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4190) // C-linkage returning std::complex<> (OpenBLAS lapacke.h)
#endif
#include <lapacke.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "include/common/workspace_arena.h"

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
static void check_symmetric_eigen_inputs(const void* A,
    const void*                                      eigenvalues,
    const void*                                      eigenvectors,
    linalg_int                                       n,
    linalg_int                                       lda,
    linalg_int                                       ldv)
{
    LINALG_CHECK(A != nullptr && eigenvalues != nullptr && eigenvectors != nullptr,
        "symmetric_eigen_backend::apply: A, eigenvalues, and eigenvectors must not be null");
    LINALG_CHECK(
        n > 0 && lda > 0 && ldv > 0, "symmetric_eigen_backend::apply: n/lda/ldv must be positive");
}

static void check_general_eigen_inputs(const void* A,
    const void*                                    eigenvalues_real,
    const void*                                    eigenvalues_imag,
    const void*                                    eigenvectors,
    linalg_int                                     n,
    linalg_int                                     lda,
    linalg_int                                     ldv)
{
    LINALG_CHECK(A != nullptr && eigenvalues_real != nullptr && eigenvalues_imag != nullptr,
        "general_eigen_backend::apply: A, eigenvalues_real, and eigenvalues_imag must not be null");
    LINALG_CHECK(n > 0 && lda > 0 && !(eigenvectors != nullptr && ldv <= 0),
        "general_eigen_backend::apply: n/lda/ldv must be positive");
}

}  // namespace detail

namespace
{
template <typename T> size_t symmetric_eigen_blas_workspace_size(linalg_int n)
{
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return arena.total_size();
}

template <typename T>
size_t general_eigen_blas_workspace_size(linalg_int n, bool compute_eigenvectors)
{
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    if (compute_eigenvectors)
    {
        arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    }
    return arena.total_size();
}
}  // namespace

template <>
bool symmetric_eigen_backend<float, linalg_backend::blas_lapacke>::apply(const float* A,
    linalg_int                                                                        n,
    linalg_int                                                                        lda,
    float*                                                                            eigenvalues,
    float*                                                                            eigenvectors,
    linalg_int                                                                        ldv,
    void*                                                                             workspace,
    size_t workspace_size)
{
    LINALG_SCOPE("eigen/blas_lapacke<float>");
    detail::check_symmetric_eigen_inputs(A, eigenvalues, eigenvectors, n, lda, ldv);
    LINALG_CHECK(workspace_size >= symmetric_eigen_blas_workspace_size<float>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*      buf = arena.next<float>(static_cast<size_t>(n) * static_cast<size_t>(n));
    const auto nn  = static_cast<lapack_int>(n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, static_cast<std::size_t>(n), buf + i * n);
    }
    auto info = LAPACKE_ssyevd(LAPACK_ROW_MAJOR, 'V', 'L', nn, buf, nn, eigenvalues);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(buf + i * n, static_cast<std::size_t>(n), eigenvectors + i * ldv);
    }
    return info == 0;
}

template <>
bool symmetric_eigen_backend<double, linalg_backend::blas_lapacke>::apply(const double* A,
    linalg_int                                                                          n,
    linalg_int                                                                          lda,
    double*                                                                             eigenvalues,
    double*    eigenvectors,
    linalg_int ldv,
    void*      workspace,
    size_t     workspace_size)
{
    LINALG_SCOPE("eigen/blas_lapacke<double>");
    detail::check_symmetric_eigen_inputs(A, eigenvalues, eigenvectors, n, lda, ldv);
    LINALG_CHECK(workspace_size >= symmetric_eigen_blas_workspace_size<double>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*      buf = arena.next<double>(static_cast<size_t>(n) * static_cast<size_t>(n));
    const auto nn  = static_cast<lapack_int>(n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, static_cast<std::size_t>(n), buf + i * n);
    }
    auto info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', nn, buf, nn, eigenvalues);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(buf + i * n, static_cast<std::size_t>(n), eigenvectors + i * ldv);
    }
    return info == 0;
}

template <>
bool general_eigen_backend<float, linalg_backend::blas_lapacke>::apply(const float* A,
    linalg_int                                                                      n,
    linalg_int                                                                      lda,
    float*     eigenvalues_real,
    float*     eigenvalues_imag,
    float*     eigenvectors,
    linalg_int ldv,
    void*      workspace,
    size_t     workspace_size)
{
    LINALG_SCOPE("general_eigen/blas_lapacke<float>");
    detail::check_general_eigen_inputs(
        A, eigenvalues_real, eigenvalues_imag, eigenvectors, n, lda, ldv);
    const bool compute_eigenvectors = eigenvectors != nullptr;
    LINALG_CHECK(
        workspace_size >= general_eigen_blas_workspace_size<float>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bytes = static_cast<size_t>(n) * static_cast<size_t>(n);
    auto*                   buf   = arena.next<float>(bytes);
    const auto              nn    = static_cast<lapack_int>(n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, static_cast<std::size_t>(n), buf + i * n);
    }
    lapack_int info = 0;
    if (compute_eigenvectors)
    {
        auto* vr = arena.next<float>(bytes);
        info     = LAPACKE_sgeev(LAPACK_ROW_MAJOR,
            'N',
            'V',
            nn,
            buf,
            nn,
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            nn,
            vr,
            nn);
        for (linalg_int i = 0; i < n; ++i)
        {
            std::copy_n(vr + i * n, static_cast<std::size_t>(n), eigenvectors + i * ldv);
        }
    }
    else
    {
        info = LAPACKE_sgeev(LAPACK_ROW_MAJOR,
            'N',
            'N',
            nn,
            buf,
            nn,
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            nn,
            nullptr,
            nn);
    }
    return info == 0;
}

template <>
bool general_eigen_backend<double, linalg_backend::blas_lapacke>::apply(const double* A,
    linalg_int                                                                        n,
    linalg_int                                                                        lda,
    double*    eigenvalues_real,
    double*    eigenvalues_imag,
    double*    eigenvectors,
    linalg_int ldv,
    void*      workspace,
    size_t     workspace_size)
{
    LINALG_SCOPE("general_eigen/blas_lapacke<double>");
    detail::check_general_eigen_inputs(
        A, eigenvalues_real, eigenvalues_imag, eigenvectors, n, lda, ldv);
    const bool compute_eigenvectors = eigenvectors != nullptr;
    LINALG_CHECK(
        workspace_size >= general_eigen_blas_workspace_size<double>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bytes = static_cast<size_t>(n) * static_cast<size_t>(n);
    auto*                   buf   = arena.next<double>(bytes);
    const auto              nn    = static_cast<lapack_int>(n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, static_cast<std::size_t>(n), buf + i * n);
    }
    lapack_int info = 0;
    if (compute_eigenvectors)
    {
        auto* vr = arena.next<double>(bytes);
        info     = LAPACKE_dgeev(LAPACK_ROW_MAJOR,
            'N',
            'V',
            nn,
            buf,
            nn,
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            nn,
            vr,
            nn);
        for (linalg_int i = 0; i < n; ++i)
        {
            std::copy_n(vr + i * n, static_cast<std::size_t>(n), eigenvectors + i * ldv);
        }
    }
    else
    {
        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,
            'N',
            'N',
            nn,
            buf,
            nn,
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            nn,
            nullptr,
            nn);
    }
    return info == 0;
}

template <>
size_t symmetric_eigen_backend<float, linalg_backend::blas_lapacke>::workspace_size(linalg_int n)
{
    return symmetric_eigen_blas_workspace_size<float>(n);
}

template <>
size_t symmetric_eigen_backend<double, linalg_backend::blas_lapacke>::workspace_size(linalg_int n)
{
    return symmetric_eigen_blas_workspace_size<double>(n);
}

template <>
size_t general_eigen_backend<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_eigen_blas_workspace_size<float>(n, compute_eigenvectors);
}

template <>
size_t general_eigen_backend<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_eigen_blas_workspace_size<double>(n, compute_eigenvectors);
}

}  // namespace linalg
#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
