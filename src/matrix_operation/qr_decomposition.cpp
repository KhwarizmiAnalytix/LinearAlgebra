#include "include/matrix_operation/qr_decomposition.h"

#include <include/logging.h>

#include "include/common/workspace_arena.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>

#include <algorithm>
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
#include <lapacke.h>

#include <algorithm>
#else
#include <algorithm>
#include <cmath>
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// qr_decomposition() overloads below call — exactly one is ever built. No
// GPU implementation exists for this op.
#if defined(LINALG_ENABLE_MKL) || (defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE))

namespace
{

// `work` is shared sequentially by geqrf and orgqr (sized to the larger of
// the two), matching the GPU transpose-scratch reuse trick.
template <typename T>
struct qr_layout
{
    T* buf;
    T* tau;
    T* work;
};

template <typename T>
qr_layout<T> qr_plan(detail::workspace_arena& arena, linalg_long rows, linalg_long columns,
    linalg_long k, lapack_int lwork)
{
    qr_layout<T> p;
    p.buf  = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.tau  = arena.next<T>(static_cast<size_t>(k));
    p.work = arena.next<T>(static_cast<size_t>(lwork));
    return p;
}

template <typename T, typename GeqrfWorkFn, typename OrgqrWorkFn>
lapack_int qr_lwork(GeqrfWorkFn geqrf_work, OrgqrWorkFn orgqr_work, lapack_int r, lapack_int c, lapack_int k)
{
    T query[1];
    geqrf_work(LAPACK_ROW_MAJOR, r, c, nullptr, c, nullptr, query, -1);
    const auto lwork_geqrf = static_cast<lapack_int>(query[0]);
    orgqr_work(LAPACK_ROW_MAJOR, r, k, k, nullptr, c, nullptr, query, -1);
    const auto lwork_orgqr = static_cast<lapack_int>(query[0]);
    return std::max(lwork_geqrf, lwork_orgqr);
}

template <typename T, typename GeqrfWorkFn, typename OrgqrWorkFn>
size_t qr_workspace_size_impl(GeqrfWorkFn geqrf_work, OrgqrWorkFn orgqr_work, linalg_long rows, linalg_long columns)
{
    const auto k = std::min(rows, columns);
    const auto r = static_cast<lapack_int>(rows);
    const auto c = static_cast<lapack_int>(columns);
    const auto kk = static_cast<lapack_int>(k);
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
    T*                            Q,
    linalg_long                  ldq,
    T*                            R,
    linalg_long                  ldr,
    void*                        workspace,
    size_t                       workspace_size)
{
    const linalg_long k = std::min(rows, columns);
    const auto         r = static_cast<lapack_int>(rows);
    const auto         c = static_cast<lapack_int>(columns);
    const auto         kk = static_cast<lapack_int>(k);
    const auto         lwork = qr_lwork<T>(geqrf_work, orgqr_work, r, c, kk);

    LOGGING_CHECK(workspace_size >= qr_workspace_size_impl<T>(geqrf_work, orgqr_work, rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = qr_plan<T>(arena, rows, columns, k, lwork);

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

#endif  // LINALG_ENABLE_MKL || (LINALG_ENABLE_BLAS && LINALG_BLAS_HAS_LAPACKE)

#if defined(LINALG_ENABLE_MKL)

size_t qr_mkl_workspace_size_f32(linalg_long rows, linalg_long columns)
{
    return qr_workspace_size_impl<float>(LAPACKE_sgeqrf_work, LAPACKE_sorgqr_work, rows, columns);
}

size_t qr_mkl_workspace_size_f64(linalg_long rows, linalg_long columns)
{
    return qr_workspace_size_impl<double>(LAPACKE_dgeqrf_work, LAPACKE_dorgqr_work, rows, columns);
}

void qr_mkl_f32(linalg_long rows, linalg_long columns, const float* A, linalg_long lda, float* Q,
    linalg_long ldq, float* R, linalg_long ldr, void* workspace, size_t workspace_size)
{
    qr_lapacke_impl<float>(LAPACKE_sgeqrf_work, LAPACKE_sorgqr_work, "mkl", rows, columns, A, lda, Q,
        ldq, R, ldr, workspace, workspace_size);
}

void qr_mkl_f64(linalg_long rows, linalg_long columns, const double* A, linalg_long lda, double* Q,
    linalg_long ldq, double* R, linalg_long ldr, void* workspace, size_t workspace_size)
{
    qr_lapacke_impl<double>(LAPACKE_dgeqrf_work, LAPACKE_dorgqr_work, "mkl", rows, columns, A, lda,
        Q, ldq, R, ldr, workspace, workspace_size);
}

#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

size_t qr_blas_workspace_size_f32(linalg_long rows, linalg_long columns)
{
    return qr_workspace_size_impl<float>(LAPACKE_sgeqrf_work, LAPACKE_sorgqr_work, rows, columns);
}

size_t qr_blas_workspace_size_f64(linalg_long rows, linalg_long columns)
{
    return qr_workspace_size_impl<double>(LAPACKE_dgeqrf_work, LAPACKE_dorgqr_work, rows, columns);
}

void qr_blas_f32(linalg_long rows, linalg_long columns, const float* A, linalg_long lda, float* Q,
    linalg_long ldq, float* R, linalg_long ldr, void* workspace, size_t workspace_size)
{
    qr_lapacke_impl<float>(LAPACKE_sgeqrf_work, LAPACKE_sorgqr_work, "blas_lapack", rows, columns, A,
        lda, Q, ldq, R, ldr, workspace, workspace_size);
}

void qr_blas_f64(linalg_long rows, linalg_long columns, const double* A, linalg_long lda, double* Q,
    linalg_long ldq, double* R, linalg_long ldr, void* workspace, size_t workspace_size)
{
    qr_lapacke_impl<double>(LAPACKE_dgeqrf_work, LAPACKE_dorgqr_work, "blas_lapack", rows, columns,
        A, lda, Q, ldq, R, ldr, workspace, workspace_size);
}

#else

namespace
{

template <typename T>
struct qr_householder_layout
{
    T* Rwork;
    T* Qwork;
    T* v;
};

template <typename T>
qr_householder_layout<T> qr_householder_plan(detail::workspace_arena& arena, linalg_long rows, linalg_long columns)
{
    qr_householder_layout<T> p;
    p.Rwork = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.Qwork = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(rows));
    p.v     = arena.next<T>(static_cast<size_t>(rows));
    return p;
}

template <typename T>
size_t qr_householder_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    qr_householder_plan<T>(arena, rows, columns);
    return arena.total_size();
}

// Householder QR: builds Q (rows x rows, accumulated as a product of
// reflectors) and an upper-triangular working copy of A, then the caller
// slices out the economy-size Q (first k columns) / R (first k rows).
template <typename T>
void qr_householder(linalg_long rows,
    linalg_long                 columns,
    const T*                      A,
    linalg_long                  lda,
    T*                             Q,
    linalg_long                  ldq,
    T*                             R,
    linalg_long                  ldr,
    void*                         workspace,
    size_t                        workspace_size)
{
    const linalg_long k = std::min(rows, columns);

    LOGGING_CHECK(workspace_size >= qr_householder_workspace_size_impl<T>(rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs  = qr_householder_plan<T>(arena, rows, columns);
    auto* const               Rwork = bufs.Rwork;
    auto* const               Qwork = bufs.Qwork;
    auto* const               v     = bufs.v;

    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            Rwork[i * columns + j] = A[i * lda + j];
        }
    }

    std::fill_n(Qwork, rows * rows, T(0));
    for (linalg_long i = 0; i < rows; ++i)
    {
        Qwork[i * rows + i] = T(1);
    }

    for (linalg_long i = 0; i < k; ++i)
    {
        T norm = T(0);
        for (linalg_long r = i; r < rows; ++r)
        {
            const T x = Rwork[r * columns + i];
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm == T(0))
        {
            continue;
        }

        const T diag  = Rwork[i * columns + i];
        const T alpha = (diag >= T(0)) ? -norm : norm;

        for (linalg_long r = i; r < rows; ++r)
        {
            v[static_cast<std::size_t>(r)] = Rwork[r * columns + i];
        }
        v[static_cast<std::size_t>(i)] -= alpha;

        T vnorm = T(0);
        for (linalg_long r = i; r < rows; ++r)
        {
            vnorm += v[static_cast<std::size_t>(r)] * v[static_cast<std::size_t>(r)];
        }
        vnorm = std::sqrt(vnorm);
        if (vnorm == T(0))
        {
            continue;
        }
        for (linalg_long r = i; r < rows; ++r)
        {
            v[static_cast<std::size_t>(r)] /= vnorm;
        }

        // Apply H = I - 2*v*v^T to Rwork's columns [i, columns) over rows [i, rows).
        for (linalg_long j = i; j < columns; ++j)
        {
            T dot = T(0);
            for (linalg_long r = i; r < rows; ++r)
            {
                dot += v[static_cast<std::size_t>(r)] * Rwork[r * columns + j];
            }
            for (linalg_long r = i; r < rows; ++r)
            {
                Rwork[r * columns + j] -= T(2) * v[static_cast<std::size_t>(r)] * dot;
            }
        }

        // Accumulate Q = Q * H over Qwork's columns [i, rows).
        for (linalg_long r = 0; r < rows; ++r)
        {
            T dot = T(0);
            for (linalg_long c = i; c < rows; ++c)
            {
                dot += Qwork[r * rows + c] * v[static_cast<std::size_t>(c)];
            }
            for (linalg_long c = i; c < rows; ++c)
            {
                Qwork[r * rows + c] -= T(2) * dot * v[static_cast<std::size_t>(c)];
            }
        }
    }

    for (linalg_long r = 0; r < rows; ++r)
    {
        for (linalg_long c = 0; c < k; ++c)
        {
            Q[r * ldq + c] = Qwork[r * rows + c];
        }
    }

    for (linalg_long r = 0; r < k; ++r)
    {
        for (linalg_long c = 0; c < columns; ++c)
        {
            R[r * ldr + c] = (c >= r) ? Rwork[r * columns + c] : T(0);
        }
    }
}

}  // namespace

size_t qr_scalar_workspace_size_f32(linalg_long rows, linalg_long columns)
{
    return qr_householder_workspace_size_impl<float>(rows, columns);
}

size_t qr_scalar_workspace_size_f64(linalg_long rows, linalg_long columns)
{
    return qr_householder_workspace_size_impl<double>(rows, columns);
}

void qr_scalar_f32(linalg_long rows,
    linalg_long                columns,
    const float*                 A,
    linalg_long                 lda,
    float*                       Q,
    linalg_long                 ldq,
    float*                       R,
    linalg_long                 ldr,
    void*                       workspace,
    size_t                      workspace_size)
{
    qr_householder(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
}

void qr_scalar_f64(linalg_long rows,
    linalg_long                columns,
    const double*                A,
    linalg_long                 lda,
    double*                      Q,
    linalg_long                 ldq,
    double*                      R,
    linalg_long                 ldr,
    void*                       workspace,
    size_t                      workspace_size)
{
    qr_householder(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
}

#endif

}  // namespace detail

size_t qr_decomposition_workspace_size(const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
#if defined(LINALG_ENABLE_MKL)
    return detail::qr_mkl_workspace_size_f32(rows, columns);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::qr_blas_workspace_size_f32(rows, columns);
#else
    return detail::qr_scalar_workspace_size_f32(rows, columns);
#endif
}

size_t qr_decomposition_workspace_size(const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
#if defined(LINALG_ENABLE_MKL)
    return detail::qr_mkl_workspace_size_f64(rows, columns);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::qr_blas_workspace_size_f64(rows, columns);
#else
    return detail::qr_scalar_workspace_size_f64(rows, columns);
#endif
}

//-----------------------------------------------------------------------------
void qr_decomposition(linalg_long rows,
    linalg_long                   columns,
    const float*                    A,
    linalg_long                   lda,
    float*                          Q,
    linalg_long                   ldq,
    float*                          R,
    linalg_long                   ldr,
    void*                           workspace,
    size_t                          workspace_size)
{
    LOGGING_CHECK(A != nullptr && Q != nullptr && R != nullptr, "qr_decomposition: A, Q, and R must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldq != 0 && ldr != 0, "qr_decomposition: rows/columns/lda/ldq/ldr must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::qr_mkl_f32(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::qr_blas_f32(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
#else
    detail::qr_scalar_f32(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
#endif
}

//-----------------------------------------------------------------------------
void qr_decomposition(linalg_long rows,
    linalg_long                   columns,
    const double*                   A,
    linalg_long                   lda,
    double*                         Q,
    linalg_long                   ldq,
    double*                         R,
    linalg_long                   ldr,
    void*                           workspace,
    size_t                          workspace_size)
{
    LOGGING_CHECK(A != nullptr && Q != nullptr && R != nullptr, "qr_decomposition: A, Q, and R must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldq != 0 && ldr != 0, "qr_decomposition: rows/columns/lda/ldq/ldr must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::qr_mkl_f64(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::qr_blas_f64(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
#else
    detail::qr_scalar_f64(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
#endif
}

}  // namespace linalg
