#if LINALG_HAS_MKL

#include "include/matrix_operation/eigen.h"

#include "include/logging.h"

#include <algorithm>
#include <mkl.h>

#include "include/common/workspace_arena.h"

// MKL specializations for eigen.h (the native backend lives in
// eigen.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
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
    LOGGING_CHECK(A != nullptr && eigenvalues != nullptr && eigenvectors != nullptr,
        "symmetric_eigen_backend::apply: A, eigenvalues, and eigenvectors must not be null");
    LOGGING_CHECK(
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
    LOGGING_CHECK(A != nullptr && eigenvalues_real != nullptr && eigenvalues_imag != nullptr,
        "general_eigen_backend::apply: A, eigenvalues_real, and eigenvalues_imag must not be null");
    LOGGING_CHECK(n > 0 && lda > 0 && !(eigenvectors != nullptr && ldv <= 0),
        "general_eigen_backend::apply: n/lda/ldv must be positive");
}

}  // namespace detail

namespace
{
template <typename T> size_t symmetric_eigen_mkl_workspace_size(linalg_int n)
{
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return arena.total_size();
}

template <typename T>
size_t general_eigen_mkl_workspace_size(linalg_int n, bool compute_eigenvectors)
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
bool symmetric_eigen_backend<float, linalg_backend::mkl>::apply(const float* A,
    linalg_int                                                               n,
    linalg_int                                                               lda,
    float*                                                                   eigenvalues,
    float*                                                                   eigenvectors,
    linalg_int                                                               ldv,
    void*                                                                    workspace,
    size_t                                                                   workspace_size)
{
    detail::check_symmetric_eigen_inputs(A, eigenvalues, eigenvectors, n, lda, ldv);
    LOGGING_CHECK(workspace_size >= symmetric_eigen_mkl_workspace_size<float>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto* buf = arena.next<float>(static_cast<size_t>(n) * static_cast<size_t>(n));
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, n, buf + i * n);
    }
    auto info = LAPACKE_ssyevd(LAPACK_ROW_MAJOR, 'V', 'L', n, buf, n, eigenvalues);  // NOLINT
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(buf + i * n, n, eigenvectors + i * ldv);
    }
    return info == 0;
}

template <>
bool symmetric_eigen_backend<double, linalg_backend::mkl>::apply(const double* A,
    linalg_int                                                                 n,
    linalg_int                                                                 lda,
    double*                                                                    eigenvalues,
    double*                                                                    eigenvectors,
    linalg_int                                                                 ldv,
    void*                                                                      workspace,
    size_t                                                                     workspace_size)
{
    detail::check_symmetric_eigen_inputs(A, eigenvalues, eigenvectors, n, lda, ldv);
    LOGGING_CHECK(workspace_size >= symmetric_eigen_mkl_workspace_size<double>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto* buf = arena.next<double>(static_cast<size_t>(n) * static_cast<size_t>(n));
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, n, buf + i * n);
    }
    auto info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', n, buf, n, eigenvalues);  // NOLINT
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(buf + i * n, n, eigenvectors + i * ldv);
    }
    return info == 0;
}

template <>
bool general_eigen_backend<float, linalg_backend::mkl>::apply(const float* A,
    linalg_int                                                             n,
    linalg_int                                                             lda,
    float*                                                                 eigenvalues_real,
    float*                                                                 eigenvalues_imag,
    float*                                                                 eigenvectors,
    linalg_int                                                             ldv,
    void*                                                                  workspace,
    size_t                                                                 workspace_size)
{
    detail::check_general_eigen_inputs(
        A, eigenvalues_real, eigenvalues_imag, eigenvectors, n, lda, ldv);
    const bool compute_eigenvectors = eigenvectors != nullptr;
    LOGGING_CHECK(
        workspace_size >= general_eigen_mkl_workspace_size<float>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              nn  = static_cast<size_t>(n) * static_cast<size_t>(n);
    auto*                   buf = arena.next<float>(nn);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, n, buf + i * n);
    }
    linalg_int info = 0;
    if (compute_eigenvectors)
    {
        auto* vr = arena.next<float>(nn);
        info     = LAPACKE_sgeev(  // NOLINT
            LAPACK_ROW_MAJOR,
            'N',
            'V',
            n,
            buf,
            n,
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            n,
            vr,
            n);
        for (linalg_int i = 0; i < n; ++i)
        {
            std::copy_n(vr + i * n, n, eigenvectors + i * ldv);
        }
    }
    else
    {
        info = LAPACKE_sgeev(  // NOLINT
            LAPACK_ROW_MAJOR,
            'N',
            'N',
            n,
            buf,
            n,
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            n,
            nullptr,
            n);
    }
    return info == 0;
}

template <>
bool general_eigen_backend<double, linalg_backend::mkl>::apply(const double* A,
    linalg_int                                                               n,
    linalg_int                                                               lda,
    double*                                                                  eigenvalues_real,
    double*                                                                  eigenvalues_imag,
    double*                                                                  eigenvectors,
    linalg_int                                                               ldv,
    void*                                                                    workspace,
    size_t                                                                   workspace_size)
{
    detail::check_general_eigen_inputs(
        A, eigenvalues_real, eigenvalues_imag, eigenvectors, n, lda, ldv);
    const bool compute_eigenvectors = eigenvectors != nullptr;
    LOGGING_CHECK(
        workspace_size >= general_eigen_mkl_workspace_size<double>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              nn  = static_cast<size_t>(n) * static_cast<size_t>(n);
    auto*                   buf = arena.next<double>(nn);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, n, buf + i * n);
    }
    linalg_int info = 0;
    if (compute_eigenvectors)
    {
        auto* vr = arena.next<double>(nn);
        info     = LAPACKE_dgeev(  // NOLINT
            LAPACK_ROW_MAJOR,
            'N',
            'V',
            n,
            buf,
            n,
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            n,
            vr,
            n);
        for (linalg_int i = 0; i < n; ++i)
        {
            std::copy_n(vr + i * n, n, eigenvectors + i * ldv);
        }
    }
    else
    {
        info = LAPACKE_dgeev(  // NOLINT
            LAPACK_ROW_MAJOR,
            'N',
            'N',
            n,
            buf,
            n,
            eigenvalues_real,
            eigenvalues_imag,
            nullptr,
            n,
            nullptr,
            n);
    }
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
