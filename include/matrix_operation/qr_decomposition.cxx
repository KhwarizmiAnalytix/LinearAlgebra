#include "include/matrix_operation/qr_decomposition.h"

#include <include/logging.h>

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>

#include <algorithm>
#include <vector>

#include "include/memory/allocator.h"
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
#include <lapacke.h>

#include <algorithm>
#include <vector>

#include "include/memory/allocator.h"
#else
#include <algorithm>
#include <cmath>
#include <vector>

#include "include/memory/allocator.h"
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// qr_decomposition() overloads below call — exactly one is ever built. No
// GPU implementation exists for this op.
#if defined(LINALG_ENABLE_MKL)

void qr_mkl_f32(linalg_long rows,
    linalg_long             columns,
    const float*              A,
    linalg_long              lda,
    float*                    Q,
    linalg_long              ldq,
    float*                    R,
    linalg_long              ldr)
{
    const linalg_long k = std::min(rows, columns);
    using Allocator       = linalg::allocator<float>;
    auto* buf             = Allocator::allocate(rows * columns);
    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(A + i * lda, columns, buf + i * columns);
    }

    std::vector<float> tau(static_cast<std::size_t>(k));
    auto info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, rows, columns, buf, columns, tau.data());  // NOLINT
    if (info != 0)
    {
        Allocator::free(buf);
        LOGGING_THROW("mkl QR (geqrf) was unsuccessful");
    }

    for (linalg_long i = 0; i < k; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            R[i * ldr + j] = (j >= i) ? buf[i * columns + j] : 0.F;
        }
    }

    info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, rows, k, k, buf, columns, tau.data());  // NOLINT
    if (info != 0)
    {
        Allocator::free(buf);
        LOGGING_THROW("mkl QR (orgqr) was unsuccessful");
    }

    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(buf + i * columns, k, Q + i * ldq);
    }

    Allocator::free(buf);
}

void qr_mkl_f64(linalg_long rows,
    linalg_long             columns,
    const double*             A,
    linalg_long              lda,
    double*                   Q,
    linalg_long              ldq,
    double*                   R,
    linalg_long              ldr)
{
    const linalg_long k = std::min(rows, columns);
    using Allocator       = linalg::allocator<double>;
    auto* buf             = Allocator::allocate(rows * columns);
    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(A + i * lda, columns, buf + i * columns);
    }

    std::vector<double> tau(static_cast<std::size_t>(k));
    auto info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows, columns, buf, columns, tau.data());  // NOLINT
    if (info != 0)
    {
        Allocator::free(buf);
        LOGGING_THROW("mkl QR (geqrf) was unsuccessful");
    }

    for (linalg_long i = 0; i < k; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            R[i * ldr + j] = (j >= i) ? buf[i * columns + j] : 0.;
        }
    }

    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows, k, k, buf, columns, tau.data());  // NOLINT
    if (info != 0)
    {
        Allocator::free(buf);
        LOGGING_THROW("mkl QR (orgqr) was unsuccessful");
    }

    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(buf + i * columns, k, Q + i * ldq);
    }

    Allocator::free(buf);
}

#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

void qr_blas_f32(linalg_long rows,
    linalg_long              columns,
    const float*               A,
    linalg_long               lda,
    float*                     Q,
    linalg_long               ldq,
    float*                     R,
    linalg_long               ldr)
{
    const linalg_long k = std::min(rows, columns);
    using Allocator       = linalg::allocator<float>;
    auto* buf             = Allocator::allocate(rows * columns);
    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(A + i * lda, columns, buf + i * columns);
    }

    const auto r = static_cast<lapack_int>(rows);
    const auto c = static_cast<lapack_int>(columns);
    const auto kk = static_cast<lapack_int>(k);

    std::vector<float> tau(static_cast<std::size_t>(k));
    auto info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, r, c, buf, c, tau.data());
    if (info != 0)
    {
        Allocator::free(buf);
        LOGGING_THROW("blas_lapack QR (geqrf) was unsuccessful");
    }

    for (linalg_long i = 0; i < k; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            R[i * ldr + j] = (j >= i) ? buf[i * columns + j] : 0.F;
        }
    }

    info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, r, kk, kk, buf, c, tau.data());
    if (info != 0)
    {
        Allocator::free(buf);
        LOGGING_THROW("blas_lapack QR (orgqr) was unsuccessful");
    }

    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(buf + i * columns, k, Q + i * ldq);
    }

    Allocator::free(buf);
}

void qr_blas_f64(linalg_long rows,
    linalg_long              columns,
    const double*              A,
    linalg_long               lda,
    double*                    Q,
    linalg_long               ldq,
    double*                    R,
    linalg_long               ldr)
{
    const linalg_long k = std::min(rows, columns);
    using Allocator       = linalg::allocator<double>;
    auto* buf             = Allocator::allocate(rows * columns);
    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(A + i * lda, columns, buf + i * columns);
    }

    const auto r  = static_cast<lapack_int>(rows);
    const auto c  = static_cast<lapack_int>(columns);
    const auto kk = static_cast<lapack_int>(k);

    std::vector<double> tau(static_cast<std::size_t>(k));
    auto info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, r, c, buf, c, tau.data());
    if (info != 0)
    {
        Allocator::free(buf);
        LOGGING_THROW("blas_lapack QR (geqrf) was unsuccessful");
    }

    for (linalg_long i = 0; i < k; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            R[i * ldr + j] = (j >= i) ? buf[i * columns + j] : 0.;
        }
    }

    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, r, kk, kk, buf, c, tau.data());
    if (info != 0)
    {
        Allocator::free(buf);
        LOGGING_THROW("blas_lapack QR (orgqr) was unsuccessful");
    }

    for (linalg_long i = 0; i < rows; ++i)
    {
        std::copy_n(buf + i * columns, k, Q + i * ldq);
    }

    Allocator::free(buf);
}

#else

namespace
{

// Householder QR: builds Q (rows x rows, accumulated as a product of
// reflectors) and an upper-triangular working copy of A, then the caller
// slices out the economy-size Q (first k columns) / R (first k rows).
template <typename T, class Allocator = linalg::allocator<T>>
void qr_householder(linalg_long rows,
    linalg_long                 columns,
    const T*                      A,
    linalg_long                  lda,
    T*                             Q,
    linalg_long                  ldq,
    T*                             R,
    linalg_long                  ldr)
{
    const linalg_long k = std::min(rows, columns);

    auto* Rwork = Allocator::allocate(rows * columns);
    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            Rwork[i * columns + j] = A[i * lda + j];
        }
    }

    auto* Qwork = Allocator::allocate(rows * rows);
    std::fill_n(Qwork, rows * rows, T(0));
    for (linalg_long i = 0; i < rows; ++i)
    {
        Qwork[i * rows + i] = T(1);
    }

    std::vector<T> v(static_cast<std::size_t>(rows));

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

    Allocator::free(Qwork);
    Allocator::free(Rwork);
}

}  // namespace

void qr_scalar_f32(linalg_long rows,
    linalg_long                columns,
    const float*                 A,
    linalg_long                 lda,
    float*                       Q,
    linalg_long                 ldq,
    float*                       R,
    linalg_long                 ldr)
{
    qr_householder(rows, columns, A, lda, Q, ldq, R, ldr);
}

void qr_scalar_f64(linalg_long rows,
    linalg_long                columns,
    const double*                A,
    linalg_long                 lda,
    double*                      Q,
    linalg_long                 ldq,
    double*                      R,
    linalg_long                 ldr)
{
    qr_householder(rows, columns, A, lda, Q, ldq, R, ldr);
}

#endif

}  // namespace detail

//-----------------------------------------------------------------------------
void qr_decomposition(linalg_long rows,
    linalg_long                   columns,
    const float*                    A,
    linalg_long                   lda,
    float*                          Q,
    linalg_long                   ldq,
    float*                          R,
    linalg_long                   ldr)
{
    LOGGING_CHECK(A != nullptr && Q != nullptr && R != nullptr, "qr_decomposition: A, Q, and R must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldq != 0 && ldr != 0, "qr_decomposition: rows/columns/lda/ldq/ldr must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::qr_mkl_f32(rows, columns, A, lda, Q, ldq, R, ldr);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::qr_blas_f32(rows, columns, A, lda, Q, ldq, R, ldr);
#else
    detail::qr_scalar_f32(rows, columns, A, lda, Q, ldq, R, ldr);
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
    linalg_long                   ldr)
{
    LOGGING_CHECK(A != nullptr && Q != nullptr && R != nullptr, "qr_decomposition: A, Q, and R must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldq != 0 && ldr != 0, "qr_decomposition: rows/columns/lda/ldq/ldr must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::qr_mkl_f64(rows, columns, A, lda, Q, ldq, R, ldr);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::qr_blas_f64(rows, columns, A, lda, Q, ldq, R, ldr);
#else
    detail::qr_scalar_f64(rows, columns, A, lda, Q, ldq, R, ldr);
#endif
}

}  // namespace linalg
