#include "include/matrix_operation/eigenvalue_decomposition.h"

#include "include/util/exception.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>

#include <algorithm>

#include "include/memory/allocator.h"
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
#include <lapacke.h>

#include <algorithm>

#include "include/memory/allocator.h"
#else
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "include/memory/allocator.h"
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// symmetric_eigenvalue_decomposition() / eigenvalue_decomposition()
// overloads below call — exactly one is ever built. No GPU implementation
// exists for either op.
#if defined(LINALG_ENABLE_MKL)

bool symmetric_eigen_mkl_f32(
    const float* A, linalg_int n, linalg_int lda, float* eigenvalues, float* eigenvectors, linalg_int ldv)
{
    using Allocator = linalg::allocator<float>;
    auto* buf       = Allocator::allocate(n * n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, n, buf + i * n);
    }
    auto info = LAPACKE_ssyevd(LAPACK_ROW_MAJOR, 'V', 'L', n, buf, n, eigenvalues);  // NOLINT
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(buf + i * n, n, eigenvectors + i * ldv);
    }
    Allocator::free(buf);
    return info == 0;
}

bool symmetric_eigen_mkl_f64(
    const double* A, linalg_int n, linalg_int lda, double* eigenvalues, double* eigenvectors, linalg_int ldv)
{
    using Allocator = linalg::allocator<double>;
    auto* buf       = Allocator::allocate(n * n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, n, buf + i * n);
    }
    auto info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', n, buf, n, eigenvalues);  // NOLINT
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(buf + i * n, n, eigenvectors + i * ldv);
    }
    Allocator::free(buf);
    return info == 0;
}

bool general_eigen_mkl_f32(const float* A,
    linalg_int                        n,
    linalg_int                        lda,
    float*                              eigenvalues_real,
    float*                              eigenvalues_imag,
    float*                              eigenvectors,
    linalg_int                        ldv)
{
    using Allocator = linalg::allocator<float>;
    auto* buf       = Allocator::allocate(n * n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, n, buf + i * n);
    }
    linalg_int info = 0;
    if (eigenvectors != nullptr)
    {
        auto* vr = Allocator::allocate(n * n);
        info = LAPACKE_sgeev(  // NOLINT
            LAPACK_ROW_MAJOR, 'N', 'V', n, buf, n, eigenvalues_real, eigenvalues_imag, nullptr, n, vr, n);
        for (linalg_int i = 0; i < n; ++i)
        {
            std::copy_n(vr + i * n, n, eigenvectors + i * ldv);
        }
        Allocator::free(vr);
    }
    else
    {
        info = LAPACKE_sgeev(  // NOLINT
            LAPACK_ROW_MAJOR, 'N', 'N', n, buf, n, eigenvalues_real, eigenvalues_imag, nullptr, n, nullptr, n);
    }
    Allocator::free(buf);
    return info == 0;
}

bool general_eigen_mkl_f64(const double* A,
    linalg_int                         n,
    linalg_int                         lda,
    double*                              eigenvalues_real,
    double*                              eigenvalues_imag,
    double*                              eigenvectors,
    linalg_int                         ldv)
{
    using Allocator = linalg::allocator<double>;
    auto* buf       = Allocator::allocate(n * n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, n, buf + i * n);
    }
    linalg_int info = 0;
    if (eigenvectors != nullptr)
    {
        auto* vr = Allocator::allocate(n * n);
        info = LAPACKE_dgeev(  // NOLINT
            LAPACK_ROW_MAJOR, 'N', 'V', n, buf, n, eigenvalues_real, eigenvalues_imag, nullptr, n, vr, n);
        for (linalg_int i = 0; i < n; ++i)
        {
            std::copy_n(vr + i * n, n, eigenvectors + i * ldv);
        }
        Allocator::free(vr);
    }
    else
    {
        info = LAPACKE_dgeev(  // NOLINT
            LAPACK_ROW_MAJOR, 'N', 'N', n, buf, n, eigenvalues_real, eigenvalues_imag, nullptr, n, nullptr, n);
    }
    Allocator::free(buf);
    return info == 0;
}

#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

bool symmetric_eigen_blas_f32(
    const float* A, linalg_int n, linalg_int lda, float* eigenvalues, float* eigenvectors, linalg_int ldv)
{
    using Allocator = linalg::allocator<float>;
    auto*      buf  = Allocator::allocate(n * n);
    const auto nn    = static_cast<lapack_int>(n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, static_cast<std::size_t>(n), buf + i * n);
    }
    auto info = LAPACKE_ssyevd(LAPACK_ROW_MAJOR, 'V', 'L', nn, buf, nn, eigenvalues);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(buf + i * n, static_cast<std::size_t>(n), eigenvectors + i * ldv);
    }
    Allocator::free(buf);
    return info == 0;
}

bool symmetric_eigen_blas_f64(
    const double* A, linalg_int n, linalg_int lda, double* eigenvalues, double* eigenvectors, linalg_int ldv)
{
    using Allocator = linalg::allocator<double>;
    auto*      buf  = Allocator::allocate(n * n);
    const auto nn    = static_cast<lapack_int>(n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, static_cast<std::size_t>(n), buf + i * n);
    }
    auto info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', nn, buf, nn, eigenvalues);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(buf + i * n, static_cast<std::size_t>(n), eigenvectors + i * ldv);
    }
    Allocator::free(buf);
    return info == 0;
}

bool general_eigen_blas_f32(const float* A,
    linalg_int                         n,
    linalg_int                         lda,
    float*                               eigenvalues_real,
    float*                               eigenvalues_imag,
    float*                               eigenvectors,
    linalg_int                         ldv)
{
    using Allocator = linalg::allocator<float>;
    auto*      buf  = Allocator::allocate(n * n);
    const auto nn    = static_cast<lapack_int>(n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, static_cast<std::size_t>(n), buf + i * n);
    }
    lapack_int info = 0;
    if (eigenvectors != nullptr)
    {
        auto* vr = Allocator::allocate(n * n);
        info     = LAPACKE_sgeev(
            LAPACK_ROW_MAJOR, 'N', 'V', nn, buf, nn, eigenvalues_real, eigenvalues_imag, nullptr, nn, vr, nn);
        for (linalg_int i = 0; i < n; ++i)
        {
            std::copy_n(vr + i * n, static_cast<std::size_t>(n), eigenvectors + i * ldv);
        }
        Allocator::free(vr);
    }
    else
    {
        info = LAPACKE_sgeev(
            LAPACK_ROW_MAJOR, 'N', 'N', nn, buf, nn, eigenvalues_real, eigenvalues_imag, nullptr, nn, nullptr, nn);
    }
    Allocator::free(buf);
    return info == 0;
}

bool general_eigen_blas_f64(const double* A,
    linalg_int                          n,
    linalg_int                          lda,
    double*                               eigenvalues_real,
    double*                               eigenvalues_imag,
    double*                               eigenvectors,
    linalg_int                          ldv)
{
    using Allocator = linalg::allocator<double>;
    auto*      buf  = Allocator::allocate(n * n);
    const auto nn    = static_cast<lapack_int>(n);
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(A + i * lda, static_cast<std::size_t>(n), buf + i * n);
    }
    lapack_int info = 0;
    if (eigenvectors != nullptr)
    {
        auto* vr = Allocator::allocate(n * n);
        info     = LAPACKE_dgeev(
            LAPACK_ROW_MAJOR, 'N', 'V', nn, buf, nn, eigenvalues_real, eigenvalues_imag, nullptr, nn, vr, nn);
        for (linalg_int i = 0; i < n; ++i)
        {
            std::copy_n(vr + i * n, static_cast<std::size_t>(n), eigenvectors + i * ldv);
        }
        Allocator::free(vr);
    }
    else
    {
        info = LAPACKE_dgeev(
            LAPACK_ROW_MAJOR, 'N', 'N', nn, buf, nn, eigenvalues_real, eigenvalues_imag, nullptr, nn, nullptr, nn);
    }
    Allocator::free(buf);
    return info == 0;
}

#else

namespace
{

// Classic cyclic Jacobi eigenvalue algorithm (Golub & Van Loan, "Matrix
// Computations", 8.4). Diagonalizes a symmetric matrix by a sequence of
// plane rotations that each zero one off-diagonal pair; converges
// quadratically, so a modest sweep cap is plenty even though this loops
// unconditionally over all n*(n-1)/2 pairs per sweep (no work-queue /
// threshold-skip bookkeeping, favoring a simple, obviously-correct
// portable fallback over the fastest possible one).
template <typename T, class Allocator = linalg::allocator<T>>
bool jacobi_eigen(const T* A, linalg_int n, linalg_int lda, T* eigenvalues, T* V, linalg_int ldv)
{
    std::vector<T> a(static_cast<std::size_t>(n * n));
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            // Only the lower triangle of A is guaranteed meaningful (see
            // header); mirror it so the working copy is exactly symmetric.
            a[static_cast<std::size_t>(i * n + j)] = (i >= j) ? A[i * lda + j] : A[j * lda + i];
        }
    }

    std::vector<T> v(static_cast<std::size_t>(n * n), T(0));
    for (linalg_int i = 0; i < n; ++i)
    {
        v[static_cast<std::size_t>(i * n + i)] = T(1);
    }

    T scale2 = T(0);
    for (linalg_int i = 0; i < n * n; ++i)
    {
        scale2 += a[static_cast<std::size_t>(i)] * a[static_cast<std::size_t>(i)];
    }
    const T eps = std::numeric_limits<T>::epsilon();

    const int max_sweeps = 100;
    bool      converged  = (n <= 1);
    for (int sweep = 0; sweep < max_sweeps && !converged; ++sweep)
    {
        T off = T(0);
        for (linalg_int p = 0; p < n; ++p)
        {
            for (linalg_int q = p + 1; q < n; ++q)
            {
                const T apq = a[static_cast<std::size_t>(p * n + q)];
                off += apq * apq;
            }
        }
        if (off <= eps * eps * std::max(scale2, T(1)))
        {
            converged = true;
            break;
        }

        for (linalg_int p = 0; p < n - 1; ++p)
        {
            for (linalg_int q = p + 1; q < n; ++q)
            {
                const T apq = a[static_cast<std::size_t>(p * n + q)];
                if (std::fabs(apq) <= eps * eps * std::max(scale2, T(1)))
                {
                    continue;
                }

                const T app   = a[static_cast<std::size_t>(p * n + p)];
                const T aqq   = a[static_cast<std::size_t>(q * n + q)];
                const T theta = (aqq - app) / (T(2) * apq);
                T       t;
                if (theta == T(0))
                {
                    t = T(1);
                }
                else
                {
                    const T sign_theta = (theta > T(0)) ? T(1) : T(-1);
                    t                  = sign_theta / (std::fabs(theta) + std::sqrt(theta * theta + T(1)));
                }
                const T c = T(1) / std::sqrt(t * t + T(1));
                const T s = t * c;

                a[static_cast<std::size_t>(p * n + p)] = app - t * apq;
                a[static_cast<std::size_t>(q * n + q)] = aqq + t * apq;
                a[static_cast<std::size_t>(p * n + q)] = T(0);
                a[static_cast<std::size_t>(q * n + p)] = T(0);

                for (linalg_int i = 0; i < n; ++i)
                {
                    if (i == p || i == q)
                    {
                        continue;
                    }
                    const T aip                            = a[static_cast<std::size_t>(i * n + p)];
                    const T aiq                            = a[static_cast<std::size_t>(i * n + q)];
                    a[static_cast<std::size_t>(i * n + p)] = c * aip - s * aiq;
                    a[static_cast<std::size_t>(p * n + i)] = a[static_cast<std::size_t>(i * n + p)];
                    a[static_cast<std::size_t>(i * n + q)] = s * aip + c * aiq;
                    a[static_cast<std::size_t>(q * n + i)] = a[static_cast<std::size_t>(i * n + q)];
                }

                for (linalg_int i = 0; i < n; ++i)
                {
                    const T vip                            = v[static_cast<std::size_t>(i * n + p)];
                    const T viq                            = v[static_cast<std::size_t>(i * n + q)];
                    v[static_cast<std::size_t>(i * n + p)] = c * vip - s * viq;
                    v[static_cast<std::size_t>(i * n + q)] = s * vip + c * viq;
                }
            }
        }
    }

    std::vector<T> evals(static_cast<std::size_t>(n));
    for (linalg_int i = 0; i < n; ++i)
    {
        evals[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i * n + i)];
    }

    std::vector<linalg_int> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), linalg_int(0));
    std::sort(idx.begin(), idx.end(), [&](linalg_int lhs, linalg_int rhs)
        { return evals[static_cast<std::size_t>(lhs)] < evals[static_cast<std::size_t>(rhs)]; });

    for (linalg_int c = 0; c < n; ++c)
    {
        eigenvalues[c] = evals[static_cast<std::size_t>(idx[static_cast<std::size_t>(c)])];
        for (linalg_int r = 0; r < n; ++r)
        {
            V[r * ldv + c] = v[static_cast<std::size_t>(r * n + idx[static_cast<std::size_t>(c)])];
        }
    }

    return converged;
}

// Square (full) Householder QR: A (n x n) = Q * R, Q orthogonal, R upper
// triangular. Self-contained copy of the same algorithm as
// qr_decomposition.cxx's scalar fallback (kept private to this translation
// unit, matching this repo's convention of each op's .cxx being
// self-contained rather than reaching across files into another op's
// anonymous-namespace helpers) — used internally by the general eigenvalue
// QR algorithm below.
template <typename T, class Allocator = linalg::allocator<T>>
void square_qr(linalg_int n, const T* A, T* Q, T* R)
{
    auto* Rwork = Allocator::allocate(n * n);
    std::copy_n(A, static_cast<std::size_t>(n * n), Rwork);

    auto* Qwork = Allocator::allocate(n * n);
    std::fill_n(Qwork, n * n, T(0));
    for (linalg_int i = 0; i < n; ++i)
    {
        Qwork[i * n + i] = T(1);
    }

    std::vector<T> v(static_cast<std::size_t>(n));

    for (linalg_int i = 0; i < n; ++i)
    {
        T norm = T(0);
        for (linalg_int r = i; r < n; ++r)
        {
            const T x = Rwork[r * n + i];
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm == T(0))
        {
            continue;
        }

        const T diag  = Rwork[i * n + i];
        const T alpha = (diag >= T(0)) ? -norm : norm;

        for (linalg_int r = i; r < n; ++r)
        {
            v[static_cast<std::size_t>(r)] = Rwork[r * n + i];
        }
        v[static_cast<std::size_t>(i)] -= alpha;

        T vnorm = T(0);
        for (linalg_int r = i; r < n; ++r)
        {
            vnorm += v[static_cast<std::size_t>(r)] * v[static_cast<std::size_t>(r)];
        }
        vnorm = std::sqrt(vnorm);
        if (vnorm == T(0))
        {
            continue;
        }
        for (linalg_int r = i; r < n; ++r)
        {
            v[static_cast<std::size_t>(r)] /= vnorm;
        }

        for (linalg_int j = i; j < n; ++j)
        {
            T dot = T(0);
            for (linalg_int r = i; r < n; ++r)
            {
                dot += v[static_cast<std::size_t>(r)] * Rwork[r * n + j];
            }
            for (linalg_int r = i; r < n; ++r)
            {
                Rwork[r * n + j] -= T(2) * v[static_cast<std::size_t>(r)] * dot;
            }
        }

        for (linalg_int r = 0; r < n; ++r)
        {
            T dot = T(0);
            for (linalg_int c = i; c < n; ++c)
            {
                dot += Qwork[r * n + c] * v[static_cast<std::size_t>(c)];
            }
            for (linalg_int c = i; c < n; ++c)
            {
                Qwork[r * n + c] -= T(2) * dot * v[static_cast<std::size_t>(c)];
            }
        }
    }

    std::copy_n(Qwork, static_cast<std::size_t>(n * n), Q);
    for (linalg_int r = 0; r < n; ++r)
    {
        for (linalg_int c = 0; c < n; ++c)
        {
            R[r * n + c] = (c >= r) ? Rwork[r * n + c] : T(0);
        }
    }

    Allocator::free(Qwork);
    Allocator::free(Rwork);
}

// Real eigenvalues (and, via a real Schur 2x2 diagonal block, a
// complex-conjugate pair's real/imaginary parts) of a general real square
// matrix, via the explicit double-shift QR algorithm (Golub & Van Loan,
// "Matrix Computations", 7.5): only the trailing subdiagonal entry of the
// active m x m leading block is checked for deflation each iteration
// (rather than also scanning for interior negligible subdiagonals to split
// off independent blocks early) — this converges more slowly than a
// production implementation but is far simpler to get right, matching this
// file's "correct portable fallback over fastest possible one" approach
// elsewhere. No Hessenberg reduction either, for the same reason: forming
// H^2 for the shift each iteration already costs the same O(n^3) that
// Hessenberg reduction would otherwise save on the QR-factorization step.
template <typename T, class Allocator = linalg::allocator<T>>
bool general_eigenvalues_qr(linalg_int n, const T* A, linalg_int lda, T* eigen_real, T* eigen_imag)
{
    std::vector<T> H(static_cast<std::size_t>(n * n));
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            H[static_cast<std::size_t>(i * n + j)] = A[i * lda + j];
        }
    }

    const T   eps           = std::numeric_limits<T>::epsilon();
    const int max_total_iter = 30 * n * n + 100;
    int       iter           = 0;
    linalg_int m            = n;

    while (m > 0)
    {
        if (m == 1)
        {
            eigen_real[0] = H[0];
            eigen_imag[0] = T(0);
            m             = 0;
            break;
        }

        auto solve_2x2 = [&](T a, T b, T c, T d, linalg_int at)
        {
            const T tr   = a + d;
            const T det  = a * d - b * c;
            const T disc = tr * tr - T(4) * det;
            if (disc >= T(0))
            {
                const T s        = std::sqrt(disc);
                eigen_real[at]   = (tr + s) / T(2);
                eigen_imag[at]   = T(0);
                eigen_real[at + 1] = (tr - s) / T(2);
                eigen_imag[at + 1] = T(0);
            }
            else
            {
                const T s          = std::sqrt(-disc);
                eigen_real[at]     = tr / T(2);
                eigen_imag[at]     = s / T(2);
                eigen_real[at + 1] = tr / T(2);
                eigen_imag[at + 1] = -s / T(2);
            }
        };

        const T sub   = std::fabs(H[static_cast<std::size_t>((m - 1) * n + (m - 2))]);
        T       scale = std::fabs(H[static_cast<std::size_t>((m - 2) * n + (m - 2))]) +
            std::fabs(H[static_cast<std::size_t>((m - 1) * n + (m - 1))]);
        if (scale == T(0))
        {
            scale = T(1);
        }
        if (sub <= eps * scale)
        {
            eigen_real[m - 1] = H[static_cast<std::size_t>((m - 1) * n + (m - 1))];
            eigen_imag[m - 1] = T(0);
            m -= 1;
            continue;
        }

        if (m == 2)
        {
            solve_2x2(H[0], H[1], H[static_cast<std::size_t>(n)], H[static_cast<std::size_t>(n + 1)], 0);
            m = 0;
            break;
        }

        const T sub2   = std::fabs(H[static_cast<std::size_t>((m - 2) * n + (m - 3))]);
        T       scale2 = std::fabs(H[static_cast<std::size_t>((m - 3) * n + (m - 3))]) +
            std::fabs(H[static_cast<std::size_t>((m - 2) * n + (m - 2))]);
        if (scale2 == T(0))
        {
            scale2 = T(1);
        }
        if (sub2 <= eps * scale2)
        {
            solve_2x2(H[static_cast<std::size_t>((m - 2) * n + (m - 2))],
                H[static_cast<std::size_t>((m - 2) * n + (m - 1))],
                H[static_cast<std::size_t>((m - 1) * n + (m - 2))],
                H[static_cast<std::size_t>((m - 1) * n + (m - 1))],
                m - 2);
            m -= 2;
            continue;
        }

        ++iter;
        if (iter > max_total_iter)
        {
            return false;
        }

        const T a  = H[static_cast<std::size_t>((m - 2) * n + (m - 2))];
        const T b  = H[static_cast<std::size_t>((m - 2) * n + (m - 1))];
        const T c  = H[static_cast<std::size_t>((m - 1) * n + (m - 2))];
        const T d  = H[static_cast<std::size_t>((m - 1) * n + (m - 1))];
        const T tr  = a + d;
        const T det = a * d - b * c;

        auto* Hm = Allocator::allocate(m * m);
        for (linalg_int i = 0; i < m; ++i)
        {
            for (linalg_int j = 0; j < m; ++j)
            {
                Hm[i * m + j] = H[static_cast<std::size_t>(i * n + j)];
            }
        }

        auto* M = Allocator::allocate(m * m);
        for (linalg_int i = 0; i < m; ++i)
        {
            for (linalg_int j = 0; j < m; ++j)
            {
                T sum = T(0);
                for (linalg_int k = 0; k < m; ++k)
                {
                    sum += Hm[i * m + k] * Hm[k * m + j];
                }
                M[i * m + j] = sum - tr * Hm[i * m + j] + ((i == j) ? det : T(0));
            }
        }

        auto* Q = Allocator::allocate(m * m);
        auto* R = Allocator::allocate(m * m);
        square_qr<T, Allocator>(m, M, Q, R);

        // Hm <- Q^T * Hm * Q
        auto* tmp = Allocator::allocate(m * m);
        for (linalg_int i = 0; i < m; ++i)
        {
            for (linalg_int j = 0; j < m; ++j)
            {
                T sum = T(0);
                for (linalg_int k = 0; k < m; ++k)
                {
                    sum += Q[k * m + i] * Hm[k * m + j];
                }
                tmp[i * m + j] = sum;
            }
        }
        for (linalg_int i = 0; i < m; ++i)
        {
            for (linalg_int j = 0; j < m; ++j)
            {
                T sum = T(0);
                for (linalg_int k = 0; k < m; ++k)
                {
                    sum += tmp[i * m + k] * Q[k * m + j];
                }
                H[static_cast<std::size_t>(i * n + j)] = sum;
            }
        }

        Allocator::free(tmp);
        Allocator::free(R);
        Allocator::free(Q);
        Allocator::free(M);
        Allocator::free(Hm);
    }

    return true;
}

// Real eigenvector via inverse iteration against the original matrix: for a
// (near-)eigenvalue lambda, (A - lambda*I) is (near-)singular, so solving
// (A - lambda*I) * x_{k+1} = x_k and renormalizing converges to the
// corresponding eigenvector. `lambda` is perturbed slightly so the LU
// factorization below never sees an exactly-singular pivot.
template <typename T, class Allocator = linalg::allocator<T>>
bool inverse_iteration(linalg_int n, const T* A, linalg_int lda, T lambda, T* eigenvector)
{
    const T eps      = std::numeric_limits<T>::epsilon();
    const T shift_eps = eps * (std::fabs(lambda) + T(1)) * T(10);

    auto* M = Allocator::allocate(n * n);
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            M[i * n + j] = A[i * lda + j] - ((i == j) ? (lambda + shift_eps) : T(0));
        }
    }

    std::vector<T> x(static_cast<std::size_t>(n), T(1) / std::sqrt(static_cast<T>(n)));
    std::vector<T> y(static_cast<std::size_t>(n));

    bool ok = true;
    for (int it = 0; it < 25 && ok; ++it)
    {
        // Solve M * y = x via plain Gaussian elimination with partial
        // pivoting (self-contained: this op does not depend on
        // lu_decomposition.cxx's scalar fallback, matching this file's
        // self-containment convention).
        auto* LU = Allocator::allocate(n * n);
        std::copy_n(M, static_cast<std::size_t>(n * n), LU);
        std::vector<T>            b(x.begin(), x.end());
        std::vector<linalg_int> piv(static_cast<std::size_t>(n));
        std::iota(piv.begin(), piv.end(), linalg_int(0));

        for (linalg_int k = 0; k < n && ok; ++k)
        {
            linalg_int max_row = k;
            T            max_val = std::fabs(LU[k * n + k]);
            for (linalg_int r = k + 1; r < n; ++r)
            {
                const T v = std::fabs(LU[r * n + k]);
                if (v > max_val)
                {
                    max_val = v;
                    max_row = r;
                }
            }
            if (max_val <= eps)
            {
                ok = false;
                break;
            }
            if (max_row != k)
            {
                for (linalg_int c = 0; c < n; ++c)
                {
                    std::swap(LU[k * n + c], LU[max_row * n + c]);
                }
                std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(max_row)]);
            }
            for (linalg_int r = k + 1; r < n; ++r)
            {
                const T f = LU[r * n + k] / LU[k * n + k];
                for (linalg_int c = k; c < n; ++c)
                {
                    LU[r * n + c] -= f * LU[k * n + c];
                }
                b[static_cast<std::size_t>(r)] -= f * b[static_cast<std::size_t>(k)];
            }
        }

        if (ok)
        {
            for (linalg_int r = n - 1; r >= 0; --r)
            {
                T sum = b[static_cast<std::size_t>(r)];
                for (linalg_int c = r + 1; c < n; ++c)
                {
                    sum -= LU[r * n + c] * y[static_cast<std::size_t>(c)];
                }
                y[static_cast<std::size_t>(r)] = sum / LU[r * n + r];
            }

            T norm = T(0);
            for (linalg_int i = 0; i < n; ++i)
            {
                norm += y[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
            }
            norm = std::sqrt(norm);
            if (norm > T(0))
            {
                for (linalg_int i = 0; i < n; ++i)
                {
                    x[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(i)] / norm;
                }
            }
        }

        Allocator::free(LU);
    }

    if (ok)
    {
        std::copy(x.begin(), x.end(), eigenvector);
    }
    Allocator::free(M);
    return ok;
}

}  // namespace

bool symmetric_eigen_scalar_f32(
    const float* A, linalg_int n, linalg_int lda, float* eigenvalues, float* eigenvectors, linalg_int ldv)
{
    return jacobi_eigen(A, n, lda, eigenvalues, eigenvectors, ldv);
}

bool symmetric_eigen_scalar_f64(
    const double* A, linalg_int n, linalg_int lda, double* eigenvalues, double* eigenvectors, linalg_int ldv)
{
    return jacobi_eigen(A, n, lda, eigenvalues, eigenvectors, ldv);
}

template <typename T>
bool general_eigen_scalar_impl(const T* A,
    linalg_int                        n,
    linalg_int                        lda,
    T*                                  eigenvalues_real,
    T*                                  eigenvalues_imag,
    T*                                  eigenvectors,
    linalg_int                        ldv)
{
    if (!general_eigenvalues_qr<T, linalg::allocator<T>>(n, A, lda, eigenvalues_real, eigenvalues_imag))
    {
        return false;
    }

    if (eigenvectors == nullptr)
    {
        return true;
    }

    const T eps = std::numeric_limits<T>::epsilon();
    for (linalg_int i = 0; i < n; ++i)
    {
        if (std::fabs(eigenvalues_imag[i]) > eps * (std::fabs(eigenvalues_real[i]) + T(1)) * T(10))
        {
            LINALG_THROW(
                "eigenvalue_decomposition: scalar fallback cannot compute eigenvectors for a complex "
                "eigenvalue (index",
                i,
                "); rebuild with LINALG_ENABLE_MKL or LINALG_ENABLE_BLAS for that case");
        }
        std::vector<T> v(static_cast<std::size_t>(n));
        if (!inverse_iteration<T, linalg::allocator<T>>(n, A, lda, eigenvalues_real[i], v.data()))
        {
            return false;
        }
        for (linalg_int r = 0; r < n; ++r)
        {
            eigenvectors[r * ldv + i] = v[static_cast<std::size_t>(r)];
        }
    }
    return true;
}

bool general_eigen_scalar_f32(const float* A,
    linalg_int                         n,
    linalg_int                         lda,
    float*                               eigenvalues_real,
    float*                               eigenvalues_imag,
    float*                               eigenvectors,
    linalg_int                         ldv)
{
    return general_eigen_scalar_impl(A, n, lda, eigenvalues_real, eigenvalues_imag, eigenvectors, ldv);
}

bool general_eigen_scalar_f64(const double* A,
    linalg_int                          n,
    linalg_int                          lda,
    double*                               eigenvalues_real,
    double*                               eigenvalues_imag,
    double*                               eigenvectors,
    linalg_int                          ldv)
{
    return general_eigen_scalar_impl(A, n, lda, eigenvalues_real, eigenvalues_imag, eigenvectors, ldv);
}

#endif

}  // namespace detail

//-----------------------------------------------------------------------------
bool symmetric_eigenvalue_decomposition(
    const float* A, linalg_int n, linalg_int lda, float* eigenvalues, float* eigenvectors, linalg_int ldv)
{
    if (A == nullptr || eigenvalues == nullptr || eigenvectors == nullptr)
    {
        LINALG_THROW("symmetric_eigenvalue_decomposition: A, eigenvalues, and eigenvectors must not be null");
    }
    if (n <= 0 || lda <= 0 || ldv <= 0)
    {
        LINALG_THROW("symmetric_eigenvalue_decomposition: n/lda/ldv must be positive");
    }
#if defined(LINALG_ENABLE_MKL)
    return detail::symmetric_eigen_mkl_f32(A, n, lda, eigenvalues, eigenvectors, ldv);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::symmetric_eigen_blas_f32(A, n, lda, eigenvalues, eigenvectors, ldv);
#else
    return detail::symmetric_eigen_scalar_f32(A, n, lda, eigenvalues, eigenvectors, ldv);
#endif
}

//-----------------------------------------------------------------------------
bool symmetric_eigenvalue_decomposition(
    const double* A, linalg_int n, linalg_int lda, double* eigenvalues, double* eigenvectors, linalg_int ldv)
{
    if (A == nullptr || eigenvalues == nullptr || eigenvectors == nullptr)
    {
        LINALG_THROW("symmetric_eigenvalue_decomposition: A, eigenvalues, and eigenvectors must not be null");
    }
    if (n <= 0 || lda <= 0 || ldv <= 0)
    {
        LINALG_THROW("symmetric_eigenvalue_decomposition: n/lda/ldv must be positive");
    }
#if defined(LINALG_ENABLE_MKL)
    return detail::symmetric_eigen_mkl_f64(A, n, lda, eigenvalues, eigenvectors, ldv);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::symmetric_eigen_blas_f64(A, n, lda, eigenvalues, eigenvectors, ldv);
#else
    return detail::symmetric_eigen_scalar_f64(A, n, lda, eigenvalues, eigenvectors, ldv);
#endif
}

//-----------------------------------------------------------------------------
bool eigenvalue_decomposition(const float* A,
    linalg_int                          n,
    linalg_int                          lda,
    float*                                eigenvalues_real,
    float*                                eigenvalues_imag,
    float*                                eigenvectors,
    linalg_int                          ldv)
{
    if (A == nullptr || eigenvalues_real == nullptr || eigenvalues_imag == nullptr)
    {
        LINALG_THROW("eigenvalue_decomposition: A, eigenvalues_real, and eigenvalues_imag must not be null");
    }
    if (n <= 0 || lda <= 0 || (eigenvectors != nullptr && ldv <= 0))
    {
        LINALG_THROW("eigenvalue_decomposition: n/lda/ldv must be positive");
    }
#if defined(LINALG_ENABLE_MKL)
    return detail::general_eigen_mkl_f32(A, n, lda, eigenvalues_real, eigenvalues_imag, eigenvectors, ldv);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::general_eigen_blas_f32(A, n, lda, eigenvalues_real, eigenvalues_imag, eigenvectors, ldv);
#else
    return detail::general_eigen_scalar_f32(A, n, lda, eigenvalues_real, eigenvalues_imag, eigenvectors, ldv);
#endif
}

//-----------------------------------------------------------------------------
bool eigenvalue_decomposition(const double* A,
    linalg_int                           n,
    linalg_int                           lda,
    double*                                eigenvalues_real,
    double*                                eigenvalues_imag,
    double*                                eigenvectors,
    linalg_int                           ldv)
{
    if (A == nullptr || eigenvalues_real == nullptr || eigenvalues_imag == nullptr)
    {
        LINALG_THROW("eigenvalue_decomposition: A, eigenvalues_real, and eigenvalues_imag must not be null");
    }
    if (n <= 0 || lda <= 0 || (eigenvectors != nullptr && ldv <= 0))
    {
        LINALG_THROW("eigenvalue_decomposition: n/lda/ldv must be positive");
    }
#if defined(LINALG_ENABLE_MKL)
    return detail::general_eigen_mkl_f64(A, n, lda, eigenvalues_real, eigenvalues_imag, eigenvectors, ldv);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::general_eigen_blas_f64(A, n, lda, eigenvalues_real, eigenvalues_imag, eigenvectors, ldv);
#else
    return detail::general_eigen_scalar_f64(A, n, lda, eigenvalues_real, eigenvalues_imag, eigenvectors, ldv);
#endif
}

}  // namespace linalg
