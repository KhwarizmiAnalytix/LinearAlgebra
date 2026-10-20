#if LINALG_HAS_MKL

#include "include/common/linalg_profiler.h"
#include "include/matrix_operation/qr_decomposition.h"
#include <mkl.h>

#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"
#include "include/common/workspace_arena.h"
#include <algorithm>

namespace linalg
{
namespace
{
// Packed A and caller-owned tau carry the factorization between phases.
template <typename T, typename Fn>
lapack_int qr_lapack_lwork(Fn fn, linalg_long rows, linalg_long columns)
{
    LINALG_CHECK(rows > 0 && columns > 0, "qr_decomposition: dimensions must be positive");
    const auto r = detail::checked_narrow<lapack_int>(rows);
    const auto c = detail::checked_narrow<lapack_int>(columns);
    T          query{};
    LINALG_CHECK(fn(r, c, &query, -1) == 0, "QR workspace query failed");
    return std::max(lapack_int(1), static_cast<lapack_int>(query));
}

template <typename T> size_t qr_lapack_size(lapack_int lwork)
{
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(lwork));
    return arena.total_size();
}

template <typename T, typename Fn>
void qr_lapack_r(Fn geqrf,
    linalg_long     rows,
    linalg_long     columns,
    T*              A,
    linalg_long     lda,
    T*              tau,
    void*           workspace,
    size_t          workspace_size,
    lapack_int      lwork)
{
    LINALG_CHECK(A && tau && workspace, "QR apply_r: A, tau, workspace required");
    LINALG_CHECK(lda >= columns, "QR apply_r: invalid row stride");
    LINALG_CHECK(workspace_size >= qr_lapack_size<T>(lwork), "QR apply_r: workspace too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   work = arena.next<T>(static_cast<size_t>(lwork));
    LINALG_CHECK(geqrf(LAPACK_ROW_MAJOR,
                     detail::checked_narrow<lapack_int>(rows),
                     detail::checked_narrow<lapack_int>(columns),
                     A,
                     detail::checked_narrow<lapack_int>(lda),
                     tau,
                     work,
                     lwork) == 0,
        "QR geqrf failed");
}

template <typename T, typename Fn>
void qr_lapack_q(Fn orgqr,
    linalg_long     rows,
    linalg_long     columns,
    T*              A,
    linalg_long     lda,
    const T*        tau,
    void*           workspace,
    size_t          workspace_size,
    lapack_int      lwork)
{
    const auto k = std::min(rows, columns);
    LINALG_CHECK(A && tau && workspace, "QR apply_q: A, tau, workspace required");
    LINALG_CHECK(lda >= columns, "QR apply_q: invalid row stride");
    LINALG_CHECK(workspace_size >= qr_lapack_size<T>(lwork), "QR apply_q: workspace too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   work = arena.next<T>(static_cast<size_t>(lwork));
    LINALG_CHECK(orgqr(LAPACK_ROW_MAJOR,
                     detail::checked_narrow<lapack_int>(rows),
                     detail::checked_narrow<lapack_int>(k),
                     detail::checked_narrow<lapack_int>(k),
                     A,
                     detail::checked_narrow<lapack_int>(lda),
                     tau,
                     work,
                     lwork) == 0,
        "QR orgqr failed");
}
}  // namespace

template <>
size_t qr_decomposition<float, linalg_backend::MKL>::workspace_size_r(
    linalg_long rows, linalg_long columns)
{
    const auto lwork =
        qr_lapack_lwork<float>([&](lapack_int r, lapack_int c, float* query, lapack_int n)
            { return LAPACKE_sgeqrf_work(LAPACK_ROW_MAJOR, r, c, nullptr, c, nullptr, query, n); },
            rows,
            columns);
    return qr_lapack_size<float>(lwork);
}
template <>
size_t qr_decomposition<float, linalg_backend::MKL>::workspace_size_q(
    linalg_long rows, linalg_long columns)
{
    const auto lwork = qr_lapack_lwork<float>(
        [&](lapack_int r, lapack_int c, float* query, lapack_int n)
        {
            const auto k = std::min(r, c);
            return LAPACKE_sorgqr_work(LAPACK_ROW_MAJOR, r, k, k, nullptr, c, nullptr, query, n);
        },
        rows,
        columns);
    return qr_lapack_size<float>(lwork);
}
template <>
void qr_decomposition<float, linalg_backend::MKL>::apply_r(linalg_long rows,
    linalg_long                                                        columns,
    float*                                                             A,
    linalg_long                                                        lda,
    float*                                                             tau,
    void*                                                              workspace,
    size_t                                                             workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/mkl<float>/apply_r");
    const auto lwork =
        qr_lapack_lwork<float>([&](lapack_int r, lapack_int c, float* query, lapack_int n)
            { return LAPACKE_sgeqrf_work(LAPACK_ROW_MAJOR, r, c, nullptr, c, nullptr, query, n); },
            rows,
            columns);
    qr_lapack_r<float>(
        LAPACKE_sgeqrf_work, rows, columns, A, lda, tau, workspace, workspace_size, lwork);
}
template <>
void qr_decomposition<float, linalg_backend::MKL>::apply_q(linalg_long rows,
    linalg_long                                                        columns,
    float*                                                             A,
    linalg_long                                                        lda,
    const float*                                                       tau,
    void*                                                              workspace,
    size_t                                                             workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/mkl<float>/apply_q");
    const auto lwork = qr_lapack_lwork<float>(
        [&](lapack_int r, lapack_int c, float* query, lapack_int n)
        {
            const auto k = std::min(r, c);
            return LAPACKE_sorgqr_work(LAPACK_ROW_MAJOR, r, k, k, nullptr, c, nullptr, query, n);
        },
        rows,
        columns);
    qr_lapack_q<float>(
        LAPACKE_sorgqr_work, rows, columns, A, lda, tau, workspace, workspace_size, lwork);
}
template <>
size_t qr_decomposition<double, linalg_backend::MKL>::workspace_size_r(
    linalg_long rows, linalg_long columns)
{
    const auto lwork =
        qr_lapack_lwork<double>([&](lapack_int r, lapack_int c, double* query, lapack_int n)
            { return LAPACKE_dgeqrf_work(LAPACK_ROW_MAJOR, r, c, nullptr, c, nullptr, query, n); },
            rows,
            columns);
    return qr_lapack_size<double>(lwork);
}
template <>
size_t qr_decomposition<double, linalg_backend::MKL>::workspace_size_q(
    linalg_long rows, linalg_long columns)
{
    const auto lwork = qr_lapack_lwork<double>(
        [&](lapack_int r, lapack_int c, double* query, lapack_int n)
        {
            const auto k = std::min(r, c);
            return LAPACKE_dorgqr_work(LAPACK_ROW_MAJOR, r, k, k, nullptr, c, nullptr, query, n);
        },
        rows,
        columns);
    return qr_lapack_size<double>(lwork);
}
template <>
void qr_decomposition<double, linalg_backend::MKL>::apply_r(linalg_long rows,
    linalg_long                                                         columns,
    double*                                                             A,
    linalg_long                                                         lda,
    double*                                                             tau,
    void*                                                               workspace,
    size_t                                                              workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/mkl<double>/apply_r");
    const auto lwork =
        qr_lapack_lwork<double>([&](lapack_int r, lapack_int c, double* query, lapack_int n)
            { return LAPACKE_dgeqrf_work(LAPACK_ROW_MAJOR, r, c, nullptr, c, nullptr, query, n); },
            rows,
            columns);
    qr_lapack_r<double>(
        LAPACKE_dgeqrf_work, rows, columns, A, lda, tau, workspace, workspace_size, lwork);
}
template <>
void qr_decomposition<double, linalg_backend::MKL>::apply_q(linalg_long rows,
    linalg_long                                                         columns,
    double*                                                             A,
    linalg_long                                                         lda,
    const double*                                                       tau,
    void*                                                               workspace,
    size_t                                                              workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/mkl<double>/apply_q");
    const auto lwork = qr_lapack_lwork<double>(
        [&](lapack_int r, lapack_int c, double* query, lapack_int n)
        {
            const auto k = std::min(r, c);
            return LAPACKE_dorgqr_work(LAPACK_ROW_MAJOR, r, k, k, nullptr, c, nullptr, query, n);
        },
        rows,
        columns);
    qr_lapack_q<double>(
        LAPACKE_dorgqr_work, rows, columns, A, lda, tau, workspace, workspace_size, lwork);
}
}  // namespace linalg
#endif
