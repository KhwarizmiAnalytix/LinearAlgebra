#include "include/matrix_operation/eigen.h"

#include "include/common/linalg_checks.h"

#include "include/common/workspace_arena.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace linalg
{
namespace detail
{

// Null/positivity checks shared by every symmetric_eigen_backend<T,
// Backend>::apply below, so callers (tests included) get the same
// validation regardless of which backend is active. There is no
// non-detail symmetric_eigenvalue_decomposition() wrapper to put this in
// any more — callers use
// symmetric_eigen_backend<T, Backend>::apply directly, naming Backend explicitly.
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

// Same, for every general_eigen_backend<T, Backend>::apply below. There is
// no non-detail eigen() wrapper to put this in any more
// — callers use general_eigen_backend<T, Backend>::apply directly, naming
// Backend explicitly.
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

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/eigenvalue_decomposition_mkl.cpp and
// blas/eigenvalue_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

template <typename T> struct jacobi_layout
{
    T*          a;
    T*          v;
    T*          evals;
    linalg_int* idx;
};

template <typename T> jacobi_layout<T> jacobi_plan(detail::workspace_arena& arena, linalg_int n)
{
    const size_t     nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    jacobi_layout<T> p;
    p.a     = arena.next<T>(nn);
    p.v     = arena.next<T>(nn);
    p.evals = arena.next<T>(static_cast<size_t>(n));
    p.idx   = arena.next<linalg_int>(static_cast<size_t>(n));
    return p;
}

template <typename T> size_t jacobi_workspace_size(linalg_int n)
{
    detail::workspace_arena arena;
    jacobi_plan<T>(arena, n);
    return arena.total_size();
}

template <typename T> struct general_scalar_layout
{
    T* H;
    T* Hm;
    T* M;
    T* Q;
    T* R;
    T* tmp;
    T* Rwork;
    T* Qwork;
    T* householder;
    T* inv_M;
    T* x;
    T* y;
    T* LU;
    T* b;
    T* evec;
};

template <typename T>
general_scalar_layout<T> general_scalar_plan(
    detail::workspace_arena& arena, linalg_int n, bool compute_eigenvectors)
{
    const size_t             nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t             ns = static_cast<size_t>(n);
    general_scalar_layout<T> p{};
    p.H           = arena.next<T>(nn);
    p.Hm          = arena.next<T>(nn);
    p.M           = arena.next<T>(nn);
    p.Q           = arena.next<T>(nn);
    p.R           = arena.next<T>(nn);
    p.tmp         = arena.next<T>(nn);
    p.Rwork       = arena.next<T>(nn);
    p.Qwork       = arena.next<T>(nn);
    p.householder = arena.next<T>(ns);
    if (compute_eigenvectors)
    {
        p.inv_M = arena.next<T>(nn);
        p.x     = arena.next<T>(ns);
        p.y     = arena.next<T>(ns);
        p.LU    = arena.next<T>(nn);
        p.b     = arena.next<T>(ns);
        p.evec  = arena.next<T>(ns);
    }
    return p;
}

template <typename T> size_t general_scalar_workspace_size(linalg_int n, bool compute_eigenvectors)
{
    detail::workspace_arena arena;
    general_scalar_plan<T>(arena, n, compute_eigenvectors);
    return arena.total_size();
}

// Classic cyclic Jacobi eigenvalue algorithm (Golub & Van Loan, "Matrix
// Computations", 8.4). Diagonalizes a symmetric matrix by a sequence of
// plane rotations that each zero one off-diagonal pair; converges
// quadratically, so a modest sweep cap is plenty even though this loops
// unconditionally over all n*(n-1)/2 pairs per sweep (no work-queue /
// threshold-skip bookkeeping, favoring a simple, obviously-correct
// portable fallback over the fastest possible one). `a`, `v`, `evals`, and
// `idx` are caller-owned scratch carved by jacobi_plan.
template <typename T>
bool jacobi_eigen(const T* A,
    linalg_int             n,
    linalg_int             lda,
    T*                     eigenvalues,
    T*                     V,
    linalg_int             ldv,
    T*                     a,
    T*                     v,
    T*                     evals,
    linalg_int*            idx)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            // Only the lower triangle of A is guaranteed meaningful (see
            // header); mirror it so the working copy is exactly symmetric.
            a[static_cast<std::size_t>(i * n + j)] = (i >= j) ? A[i * lda + j] : A[j * lda + i];
        }
    }

    std::fill_n(v, static_cast<std::size_t>(n) * static_cast<std::size_t>(n), T(0));
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
                    t = sign_theta / (std::fabs(theta) + std::sqrt(theta * theta + T(1)));
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

    for (linalg_int i = 0; i < n; ++i)
    {
        evals[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i * n + i)];
    }

    std::iota(idx, idx + n, static_cast<linalg_int>(0));
    std::sort(idx,
        idx + n,
        [&](linalg_int lhs, linalg_int rhs)
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
// qr_decomposition.cpp's scalar fallback (kept private to this translation
// unit, matching this repo's convention of each op's .cpp being
// self-contained rather than reaching across files into another op's
// anonymous-namespace helpers) — used internally by the general eigenvalue
// QR algorithm below.
template <typename T> void square_qr(linalg_int n, const T* A, T* Q, T* R, T* Rwork, T* Qwork, T* v)
{
    std::copy_n(A, static_cast<std::size_t>(n * n), Rwork);

    std::fill_n(Qwork, static_cast<std::size_t>(n) * static_cast<std::size_t>(n), T(0));
    for (linalg_int i = 0; i < n; ++i)
    {
        Qwork[i * n + i] = T(1);
    }

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
template <typename T>
bool general_eigenvalues_qr(linalg_int n,
    const T*                           A,
    linalg_int                         lda,
    T*                                 eigen_real,
    T*                                 eigen_imag,
    T*                                 H,
    T*                                 Hm,
    T*                                 M,
    T*                                 Q,
    T*                                 R,
    T*                                 tmp,
    T*                                 Rwork,
    T*                                 Qwork,
    T*                                 householder)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            H[static_cast<std::size_t>(i * n + j)] = A[i * lda + j];
        }
    }

    const T    eps            = std::numeric_limits<T>::epsilon();
    const int  max_total_iter = 30 * n * n + 100;
    int        iter           = 0;
    linalg_int m              = n;

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
                const T s          = std::sqrt(disc);
                eigen_real[at]     = (tr + s) / T(2);
                eigen_imag[at]     = T(0);
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
            solve_2x2(
                H[0], H[1], H[static_cast<std::size_t>(n)], H[static_cast<std::size_t>(n + 1)], 0);
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

        const T a   = H[static_cast<std::size_t>((m - 2) * n + (m - 2))];
        const T b   = H[static_cast<std::size_t>((m - 2) * n + (m - 1))];
        const T c   = H[static_cast<std::size_t>((m - 1) * n + (m - 2))];
        const T d   = H[static_cast<std::size_t>((m - 1) * n + (m - 1))];
        const T tr  = a + d;
        const T det = a * d - b * c;

        for (linalg_int i = 0; i < m; ++i)
        {
            for (linalg_int j = 0; j < m; ++j)
            {
                Hm[i * m + j] = H[static_cast<std::size_t>(i * n + j)];
            }
        }

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

        square_qr<T>(m, M, Q, R, Rwork, Qwork, householder);

        // Hm <- Q^T * Hm * Q
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
    }

    return true;
}

// Real eigenvector via inverse iteration against the original matrix: for a
// (near-)eigenvalue lambda, (A - lambda*I) is (near-)singular, so solving
// (A - lambda*I) * x_{k+1} = x_k and renormalizing converges to the
// corresponding eigenvector. `lambda` is perturbed slightly so the LU
// factorization below never sees an exactly-singular pivot.
template <typename T>
bool inverse_iteration(linalg_int n,
    const T*                      A,
    linalg_int                    lda,
    T                             lambda,
    T*                            eigenvector,
    T*                            M,
    T*                            x,
    T*                            y,
    T*                            LU,
    T*                            b)
{
    const T eps       = std::numeric_limits<T>::epsilon();
    const T shift_eps = eps * (std::fabs(lambda) + T(1)) * T(10);
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            M[i * n + j] = A[i * lda + j] - ((i == j) ? (lambda + shift_eps) : T(0));
        }
    }

    std::fill_n(x, static_cast<std::size_t>(n), T(1) / std::sqrt(static_cast<T>(n)));

    bool ok = true;
    for (int it = 0; it < 25 && ok; ++it)
    {
        // Solve M * y = x via plain Gaussian elimination with partial
        // pivoting (self-contained: this op does not depend on
        // lu_decomposition.cpp's scalar fallback, matching this file's
        // self-containment convention).
        std::copy_n(M, static_cast<std::size_t>(n * n), LU);
        std::copy_n(x, static_cast<std::size_t>(n), b);

        for (linalg_int k = 0; k < n && ok; ++k)
        {
            linalg_int max_row = k;
            T          max_val = std::fabs(LU[k * n + k]);
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
                std::swap(b[k], b[max_row]);
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
    }

    if (ok)
    {
        std::copy_n(x, static_cast<std::size_t>(n), eigenvector);
    }
    return ok;
}

}  // namespace

template <>
bool symmetric_eigen_backend<float, linalg_backend::native>::apply(const float* A,
    linalg_int                                                                  n,
    linalg_int                                                                  lda,
    float*                                                                      eigenvalues,
    float*                                                                      eigenvectors,
    linalg_int                                                                  ldv,
    void*                                                                       workspace,
    size_t                                                                      workspace_size)
{
    detail::check_symmetric_eigen_inputs(A, eigenvalues, eigenvectors, n, lda, ldv);
    LINALG_CHECK(workspace_size >= jacobi_workspace_size<float>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = jacobi_plan<float>(arena, n);
    return jacobi_eigen(
        A, n, lda, eigenvalues, eigenvectors, ldv, bufs.a, bufs.v, bufs.evals, bufs.idx);
}

template <>
bool symmetric_eigen_backend<double, linalg_backend::native>::apply(const double* A,
    linalg_int                                                                    n,
    linalg_int                                                                    lda,
    double*                                                                       eigenvalues,
    double*                                                                       eigenvectors,
    linalg_int                                                                    ldv,
    void*                                                                         workspace,
    size_t                                                                        workspace_size)
{
    detail::check_symmetric_eigen_inputs(A, eigenvalues, eigenvectors, n, lda, ldv);
    LINALG_CHECK(workspace_size >= jacobi_workspace_size<double>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = jacobi_plan<double>(arena, n);
    return jacobi_eigen(
        A, n, lda, eigenvalues, eigenvectors, ldv, bufs.a, bufs.v, bufs.evals, bufs.idx);
}

template <typename T>
bool general_eigen_scalar_impl(const T* A,
    linalg_int                          n,
    linalg_int                          lda,
    T*                                  eigenvalues_real,
    T*                                  eigenvalues_imag,
    T*                                  eigenvectors,
    linalg_int                          ldv,
    void*                               workspace,
    size_t                              workspace_size)
{
    const bool compute_eigenvectors = eigenvectors != nullptr;
    LINALG_CHECK(workspace_size >= general_scalar_workspace_size<T>(n, compute_eigenvectors),
        "eigen: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = general_scalar_plan<T>(arena, n, compute_eigenvectors);
    if (!general_eigenvalues_qr<T>(n,
            A,
            lda,
            eigenvalues_real,
            eigenvalues_imag,
            bufs.H,
            bufs.Hm,
            bufs.M,
            bufs.Q,
            bufs.R,
            bufs.tmp,
            bufs.Rwork,
            bufs.Qwork,
            bufs.householder))
    {
        return false;
    }

    if (!compute_eigenvectors)
    {
        return true;
    }

    const T eps = std::numeric_limits<T>::epsilon();
    for (linalg_int i = 0; i < n; ++i)
    {
        LINALG_CHECK(!(std::fabs(eigenvalues_imag[i]) >
                          eps * (std::fabs(eigenvalues_real[i]) + T(1)) * T(10)),
            "eigen: scalar fallback cannot compute eigenvectors for a complex "
            "eigenvalue (index "
            "{})",
            i);
        if (!inverse_iteration<T>(n,
                A,
                lda,
                eigenvalues_real[i],
                bufs.evec,
                bufs.inv_M,
                bufs.x,
                bufs.y,
                bufs.LU,
                bufs.b))
        {
            return false;
        }
        for (linalg_int r = 0; r < n; ++r)
        {
            eigenvectors[r * ldv + i] = bufs.evec[static_cast<std::size_t>(r)];
        }
    }
    return true;
}

template <>
bool general_eigen_backend<float, linalg_backend::native>::apply(const float* A,
    linalg_int                                                                n,
    linalg_int                                                                lda,
    float*                                                                    eigenvalues_real,
    float*                                                                    eigenvalues_imag,
    float*                                                                    eigenvectors,
    linalg_int                                                                ldv,
    void*                                                                     workspace,
    size_t                                                                    workspace_size)
{
    detail::check_general_eigen_inputs(
        A, eigenvalues_real, eigenvalues_imag, eigenvectors, n, lda, ldv);
    return general_eigen_scalar_impl(A,
        n,
        lda,
        eigenvalues_real,
        eigenvalues_imag,
        eigenvectors,
        ldv,
        workspace,
        workspace_size);
}

template <>
bool general_eigen_backend<double, linalg_backend::native>::apply(const double* A,
    linalg_int                                                                  n,
    linalg_int                                                                  lda,
    double*                                                                     eigenvalues_real,
    double*                                                                     eigenvalues_imag,
    double*                                                                     eigenvectors,
    linalg_int                                                                  ldv,
    void*                                                                       workspace,
    size_t                                                                      workspace_size)
{
    detail::check_general_eigen_inputs(
        A, eigenvalues_real, eigenvalues_imag, eigenvectors, n, lda, ldv);
    return general_eigen_scalar_impl(A,
        n,
        lda,
        eigenvalues_real,
        eigenvalues_imag,
        eigenvectors,
        ldv,
        workspace,
        workspace_size);
}

template <>
size_t symmetric_eigen_backend<float, linalg_backend::native>::workspace_size(linalg_int n)
{
    return jacobi_workspace_size<float>(n);
}

template <>
size_t symmetric_eigen_backend<double, linalg_backend::native>::workspace_size(linalg_int n)
{
    return jacobi_workspace_size<double>(n);
}

template <>
size_t general_eigen_backend<float, linalg_backend::native>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_scalar_workspace_size<float>(n, compute_eigenvectors);
}

template <>
size_t general_eigen_backend<double, linalg_backend::native>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_scalar_workspace_size<double>(n, compute_eigenvectors);
}
}  // namespace linalg
