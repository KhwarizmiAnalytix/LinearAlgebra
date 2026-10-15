#include "include/matrix_operation/qr_decomposition.h"

#include <include/logging.h>

#include "include/common/workspace_arena.h"

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
#include <lapacke.h>
#endif
#include <algorithm>

// BLAS/LAPACKE specializations for qr_decomposition.h (the native backend lives in
// qr_decomposition.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
// on; the #if below is kept as a second guard.
namespace linalg
{
#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

// Same checks as qr_decomposition.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_qr_inputs(const void* A,
    const void*                         Q,
    const void*                         R,
    linalg_long                         rows,
    linalg_long                         columns,
    linalg_long                         lda,
    linalg_long                         ldq,
    linalg_long                         ldr)
{
    LOGGING_CHECK(A != nullptr && Q != nullptr && R != nullptr,
        "qr_backend::apply: A, Q, and R must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldq != 0 && ldr != 0,
        "qr_backend::apply: rows/columns/lda/ldq/ldr must be positive");
}

namespace
{

// `work` is shared sequentially by geqrf and orgqr (sized to the larger of
// the two), matching the GPU transpose-scratch reuse trick.
template <typename T> struct qr_layout
{
    T* buf;
    T* tau;
    T* work;
};

template <typename T>
qr_layout<T> qr_plan(detail::workspace_arena& arena,
    linalg_long                               rows,
    linalg_long                               columns,
    linalg_long                               k,
    lapack_int                                lwork)
{
    qr_layout<T> p;
    p.buf  = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.tau  = arena.next<T>(static_cast<size_t>(k));
    p.work = arena.next<T>(static_cast<size_t>(lwork));
    return p;
}

template <typename T, typename GeqrfWorkFn, typename OrgqrWorkFn>
lapack_int qr_lwork(
    GeqrfWorkFn geqrf_work, OrgqrWorkFn orgqr_work, lapack_int r, lapack_int c, lapack_int k)
{
    T query[1];
    geqrf_work(LAPACK_ROW_MAJOR, r, c, nullptr, c, nullptr, query, -1);
    const auto lwork_geqrf = static_cast<lapack_int>(query[0]);
    orgqr_work(LAPACK_ROW_MAJOR, r, k, k, nullptr, c, nullptr, query, -1);
    const auto lwork_orgqr = static_cast<lapack_int>(query[0]);
    return std::max(lwork_geqrf, lwork_orgqr);
}

template <typename T, typename GeqrfWorkFn, typename OrgqrWorkFn>
size_t qr_workspace_size_impl(
    GeqrfWorkFn geqrf_work, OrgqrWorkFn orgqr_work, linalg_long rows, linalg_long columns)
{
    const auto              k  = std::min(rows, columns);
    const auto              r  = static_cast<lapack_int>(rows);
    const auto              c  = static_cast<lapack_int>(columns);
    const auto              kk = static_cast<lapack_int>(k);
    detail::workspace_arena arena;
    qr_plan<T>(arena, rows, columns, k, qr_lwork<T>(geqrf_work, orgqr_work, r, c, kk));
    return arena.total_size();
}

template <typename T, typename GeqrfWorkFn, typename OrgqrWorkFn>
void qr_lapacke_impl(GeqrfWorkFn geqrf_work,
    OrgqrWorkFn                  orgqr_work,
    const char*                  backend_name,
    linalg_long                  rows,
    linalg_long                  columns,
    const T*                     A,
    linalg_long                  lda,
    T*                           Q,
    linalg_long                  ldq,
    T*                           R,
    linalg_long                  ldr,
    void*                        workspace,
    size_t                       workspace_size)
{
    const linalg_long k     = std::min(rows, columns);
    const auto        r     = static_cast<lapack_int>(rows);
    const auto        c     = static_cast<lapack_int>(columns);
    const auto        kk    = static_cast<lapack_int>(k);
    const auto        lwork = qr_lwork<T>(geqrf_work, orgqr_work, r, c, kk);

    LOGGING_CHECK(
        workspace_size >= qr_workspace_size_impl<T>(geqrf_work, orgqr_work, rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = qr_plan<T>(arena, rows, columns, k, lwork);

    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(A + i * lda, columns, bufs.buf + i * columns);
    }

    auto info = geqrf_work(LAPACK_ROW_MAJOR, r, c, bufs.buf, c, bufs.tau, bufs.work, lwork);
    if (info != 0)
    {
        LOGGING_THROW("{} QR (geqrf) was unsuccessful", backend_name);
    }

    for (linalg_long i = 0; i < k; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            R[i * ldr + j] = (j >= i) ? bufs.buf[i * columns + j] : T(0);
        }
    }

    info = orgqr_work(LAPACK_ROW_MAJOR, r, kk, kk, bufs.buf, c, bufs.tau, bufs.work, lwork);
    if (info != 0)
    {
        LOGGING_THROW("{} QR (orgqr) was unsuccessful", backend_name);
    }

    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(bufs.buf + i * columns, static_cast<size_t>(k), Q + i * ldq);
    }
}

}  // namespace

template <>
void qr_backend<float, linalg_backend::blas_lapacke>::apply(linalg_long rows,
    linalg_long                                                         columns,
    const float*                                                        A,
    linalg_long                                                         lda,
    float*                                                              Q,
    linalg_long                                                         ldq,
    float*                                                              R,
    linalg_long                                                         ldr,
    void*                                                               workspace,
    size_t                                                              workspace_size)
{
    check_qr_inputs(A, Q, R, rows, columns, lda, ldq, ldr);
    const linalg_long k     = std::min(rows, columns);
    const auto        r     = static_cast<lapack_int>(rows);
    const auto        c     = static_cast<lapack_int>(columns);
    const auto        kk    = static_cast<lapack_int>(k);
    const auto        lwork = qr_lwork<float>(LAPACKE_sgeqrf_work, LAPACKE_sorgqr_work, r, c, kk);
    LOGGING_CHECK(workspace_size >= qr_workspace_size_impl<float>(
                                        LAPACKE_sgeqrf_work, LAPACKE_sorgqr_work, rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = qr_plan<float>(arena, rows, columns, k, lwork);
    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(A + i * lda, columns, bufs.buf + i * columns);
    }
    auto info =
        LAPACKE_sgeqrf_work(LAPACK_ROW_MAJOR, r, c, bufs.buf, c, bufs.tau, bufs.work, lwork);
    if (info != 0)
    {
        LOGGING_THROW("blas_lapack QR (geqrf) was unsuccessful");
    }
    for (linalg_long i = 0; i < k; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            R[i * ldr + j] = (j >= i) ? bufs.buf[i * columns + j] : 0.F;
        }
    }
    info =
        LAPACKE_sorgqr_work(LAPACK_ROW_MAJOR, r, kk, kk, bufs.buf, c, bufs.tau, bufs.work, lwork);
    if (info != 0)
    {
        LOGGING_THROW("blas_lapack QR (orgqr) was unsuccessful");
    }
    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(bufs.buf + i * columns, static_cast<size_t>(k), Q + i * ldq);
    }
}

template <>
void qr_backend<double, linalg_backend::blas_lapacke>::apply(linalg_long rows,
    linalg_long                                                          columns,
    const double*                                                        A,
    linalg_long                                                          lda,
    double*                                                              Q,
    linalg_long                                                          ldq,
    double*                                                              R,
    linalg_long                                                          ldr,
    void*                                                                workspace,
    size_t                                                               workspace_size)
{
    check_qr_inputs(A, Q, R, rows, columns, lda, ldq, ldr);
    const linalg_long k     = std::min(rows, columns);
    const auto        r     = static_cast<lapack_int>(rows);
    const auto        c     = static_cast<lapack_int>(columns);
    const auto        kk    = static_cast<lapack_int>(k);
    const auto        lwork = qr_lwork<double>(LAPACKE_dgeqrf_work, LAPACKE_dorgqr_work, r, c, kk);
    LOGGING_CHECK(workspace_size >= qr_workspace_size_impl<double>(
                                        LAPACKE_dgeqrf_work, LAPACKE_dorgqr_work, rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = qr_plan<double>(arena, rows, columns, k, lwork);
    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(A + i * lda, columns, bufs.buf + i * columns);
    }
    auto info =
        LAPACKE_dgeqrf_work(LAPACK_ROW_MAJOR, r, c, bufs.buf, c, bufs.tau, bufs.work, lwork);
    if (info != 0)
    {
        LOGGING_THROW("blas_lapack QR (geqrf) was unsuccessful");
    }
    for (linalg_long i = 0; i < k; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            R[i * ldr + j] = (j >= i) ? bufs.buf[i * columns + j] : 0.;
        }
    }
    info =
        LAPACKE_dorgqr_work(LAPACK_ROW_MAJOR, r, kk, kk, bufs.buf, c, bufs.tau, bufs.work, lwork);
    if (info != 0)
    {
        LOGGING_THROW("blas_lapack QR (orgqr) was unsuccessful");
    }
    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(bufs.buf + i * columns, static_cast<size_t>(k), Q + i * ldq);
    }
}

template <>
size_t qr_backend<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return qr_workspace_size_impl<float>(LAPACKE_sgeqrf_work, LAPACKE_sorgqr_work, rows, columns);
}

template <>
size_t qr_backend<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return qr_workspace_size_impl<double>(LAPACKE_dgeqrf_work, LAPACKE_dorgqr_work, rows, columns);
}

#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
}  // namespace linalg
