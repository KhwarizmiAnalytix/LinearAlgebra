#include "include/matrix_operation/eigen.h"

#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/qr_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/workspace_arena.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "include/common/linalg_profiler.h"

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

// The shifted-QR iteration runs GEMM/QR on the shrinking leading m x m block,
// and the native kernels' blocking makes their workspace need non-monotone in
// m, so size for the maximum over every m <= n rather than for m == n.
template <typename T> size_t general_scalar_mult_ws_size(linalg_int n)
{
    size_t sz = 0;
    for (linalg_int m = 1; m <= n; ++m)
    {
        sz = std::max(sz, multiplication<T, linalg_backend::native>::workspace_size(m, m, m));
    }
    return sz;
}

template <typename T> size_t general_scalar_qr_ws_size(linalg_int n)
{
    size_t sz = 0;
    for (linalg_int m = 1; m <= n; ++m)
    {
        sz = std::max(sz, qr_decomposition<T, linalg_backend::native>::workspace_size(m, m));
    }
    return sz;
}

template <typename T> struct general_scalar_layout
{
    T*             H;
    T*             Hm;
    T*             M;
    T*             Q;
    T*             R;
    T*             tmp;
    unsigned char* qr_ws;
    unsigned char* mult_ws;
    T*             inv_M;
    T*             x;
    linalg_int*    pivot;
    T*             evec;
};

template <typename T>
general_scalar_layout<T> general_scalar_plan(
    detail::workspace_arena& arena, linalg_int n, bool compute_eigenvectors)
{
    const size_t             nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t             ns = static_cast<size_t>(n);
    general_scalar_layout<T> p{};
    p.H       = arena.next<T>(nn);
    p.Hm      = arena.next<T>(nn);
    p.M       = arena.next<T>(nn);
    p.Q       = arena.next<T>(nn);
    p.R       = arena.next<T>(nn);
    p.tmp     = arena.next<T>(nn);
    p.qr_ws   = arena.next<unsigned char>(general_scalar_qr_ws_size<T>(n));
    p.mult_ws = arena.next<unsigned char>(general_scalar_mult_ws_size<T>(n));
    if (compute_eigenvectors)
    {
        p.inv_M = arena.next<T>(nn);
        p.x     = arena.next<T>(ns);
        p.pivot = arena.next<linalg_int>(ns);
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
// quadratically.
//
// Three optimisations over the naive implementation:
//  1. off-diagonal Frobenius norm `off` is maintained incrementally.  A
//     Jacobi rotation on (p,q) zeros a[p,q] and preserves the combined norm
//     of every other (i,p)+(i,q) pair (rotation is orthogonal), so the total
//     off-diagonal squared norm decreases by exactly apq² per rotation.  This
//     eliminates the O(n²) off-diagonal scan that was performed at the top of
//     every sweep.
//  2. The inner rotation update on `a` reads from rows p and q (sequential,
//     cache-friendly) rather than from columns p and q (stride-n, cache-hostile
//     on large matrices).  Writes to both the row and the transposed entry are
//     unavoidable for symmetry, but store misses are tolerated far better by
//     the CPU than load misses.
//  3. The eigenvector accumulator `vT` is stored with eigenvectors as rows
//     (vT[j*n+i] = component i of eigenvector j) so both the p-row and the
//     q-row updates are sequential memory operations.  A final O(n²) transpose
//     into V happens once after convergence.
//
// `a`, `vT`, `evals`, and `idx` are caller-owned scratch carved by jacobi_plan.
template <typename T>
bool jacobi_eigen(const T* A,
    linalg_int             n,
    linalg_int             lda,
    T*                     eigenvalues,
    T*                     V,
    linalg_int             ldv,
    T*                     a,
    T*                     vT,
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

    // vT[j*n+i] = component i of eigenvector j (rows = eigenvectors).
    std::fill_n(vT, static_cast<std::size_t>(n) * static_cast<std::size_t>(n), T(0));
    for (linalg_int i = 0; i < n; ++i)
    {
        vT[static_cast<std::size_t>(i * n + i)] = T(1);
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
        LINALG_SCOPE("eigen/jacobi/sweep");
        // Recompute the off-diagonal Frobenius norm each sweep.  This scan is
        // O(n²/2), dominated by the O(n³) rotation work, but computing it
        // from scratch avoids floating-point drift that would accumulate if it
        // were maintained incrementally across many subtractions.
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

                // Read from rows p and q (sequential) rather than columns p
                // and q (stride-n).  Write back to the row entry and its
                // symmetric counterpart to maintain the full symmetric copy.
                for (linalg_int i = 0; i < n; ++i)
                {
                    if (i == p || i == q)
                    {
                        continue;
                    }
                    const T api     = a[static_cast<std::size_t>(p * n + i)];  // row p — sequential
                    const T aqi     = a[static_cast<std::size_t>(q * n + i)];  // row q — sequential
                    const T new_api = c * api - s * aqi;
                    const T new_aqi = s * api + c * aqi;
                    a[static_cast<std::size_t>(p * n + i)] = new_api;  // row p write
                    a[static_cast<std::size_t>(i * n + p)] = new_api;  // col p write (strided)
                    a[static_cast<std::size_t>(q * n + i)] = new_aqi;  // row q write
                    a[static_cast<std::size_t>(i * n + q)] = new_aqi;  // col q write (strided)
                }

                // vT rows p and q are contiguous; both loops are sequential.
                for (linalg_int i = 0; i < n; ++i)
                {
                    const T vpi                             = vT[static_cast<std::size_t>(p * n + i)];
                    const T vqi                             = vT[static_cast<std::size_t>(q * n + i)];
                    vT[static_cast<std::size_t>(p * n + i)] = c * vpi - s * vqi;
                    vT[static_cast<std::size_t>(q * n + i)] = s * vpi + c * vqi;
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

    // Copy from vT (eigenvectors as rows) into V (eigenvectors as columns),
    // reordering by sorted eigenvalue index.
    for (linalg_int c = 0; c < n; ++c)
    {
        eigenvalues[c]               = evals[static_cast<std::size_t>(idx[static_cast<std::size_t>(c)])];
        const linalg_int src         = idx[static_cast<std::size_t>(c)];
        for (linalg_int r = 0; r < n; ++r)
        {
            V[r * ldv + c] = vT[static_cast<std::size_t>(src * n + r)];
        }
    }

    return converged;
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
    unsigned char*                     qr_ws,
    unsigned char*                     mult_ws)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            H[static_cast<std::size_t>(i * n + j)] = A[i * lda + j];
        }
    }

    const T      eps            = std::numeric_limits<T>::epsilon();
    const int    max_total_iter = 30 * n * n + 100;
    int          iter           = 0;
    linalg_int   m              = n;
    const size_t qr_ws_size     = general_scalar_qr_ws_size<T>(n);
    const size_t mult_ws_size   = general_scalar_mult_ws_size<T>(n);

    // hnorm is the Frobenius norm of the current active m×m leading block.
    // An orthogonal similarity Q^T H Q preserves the Frobenius norm, so hnorm
    // is invariant across QR steps at the same deflation level.  Only recompute
    // it when m decreases (i.e. when deflation removes a row/column).
    T          hnorm  = T(0);
    linalg_int prev_m = -1;

    while (m > 0)
    {
        LINALG_SCOPE("general_eigen/qr-iter");
        if (m == 1)
        {
            eigen_real[0] = H[0];
            eigen_imag[0] = T(0);
            m             = 0;
            break;
        }

        auto solve_2x2 = [&](T a, T b, T c, T d, linalg_int at)
        {
            // Half-difference form: tr^2 - 4*det cancels catastrophically for a
            // near-degenerate real pair (float roundoff turns it into a spurious
            // complex pair), while ((a-d)/2)^2 + b*c does not.
            const T mid  = (a + d) / T(2);
            const T p    = (a - d) / T(2);
            const T disc = p * p + b * c;
            if (disc >= T(0))
            {
                const T s          = std::sqrt(disc);
                eigen_real[at]     = mid + s;
                eigen_imag[at]     = T(0);
                eigen_real[at + 1] = mid - s;
                eigen_imag[at + 1] = T(0);
            }
            else
            {
                const T s          = std::sqrt(-disc);
                eigen_real[at]     = mid;
                eigen_imag[at]     = s;
                eigen_real[at + 1] = mid;
                eigen_imag[at + 1] = -s;
            }
        };

        // H is dense (no Hessenberg reduction), so deflation needs the whole
        // off-diagonal part of the trailing row negligible, not just its
        // subdiagonal entry, and the tolerance must be relative to the active
        // block's norm: roundoff in Q^T*H*Q leaves ~eps*||H|| residue in the
        // entries being driven to zero, which a per-entry eps*(|a|+|d|)
        // threshold can sit below in float, so the iteration never deflates.
        // Recompute hnorm only when the block size changes; it is invariant
        // under QR similarity steps at the same deflation level.
        if (m != prev_m)
        {
            T h2 = T(0);
            for (linalg_int i = 0; i < m; ++i)
            {
                for (linalg_int j = 0; j < m; ++j)
                {
                    const T x = H[static_cast<std::size_t>(i * n + j)];
                    h2 += x * x;
                }
            }
            hnorm  = std::sqrt(h2);
            if (hnorm == T(0))
            {
                hnorm = T(1);
            }
            prev_m = m;
        }
        const T tol = T(4) * eps * hnorm;

        auto row_negligible = [&](linalg_int row, linalg_int cols)
        {
            for (linalg_int j = 0; j < cols; ++j)
            {
                if (std::fabs(H[static_cast<std::size_t>(row * n + j)]) > tol)
                {
                    return false;
                }
            }
            return true;
        };

        if (row_negligible(m - 1, m - 1))
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

        if (row_negligible(m - 1, m - 2) && row_negligible(m - 2, m - 2))
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

        {
            LINALG_SCOPE("general_eigen/shift-poly");
            for (linalg_int i = 0; i < m; ++i)
            {
                for (linalg_int j = 0; j < m; ++j)
                {
                    Hm[i * m + j] = H[static_cast<std::size_t>(i * n + j)];
                }
            }

            // M <- Hm*Hm - tr*Hm + det*I, the implicit double-shift polynomial, via the
            // same native GEMM entry point multiplication.cpp exposes rather than a
            // hand-rolled triple loop.
            multiplication<T, linalg_backend::native>::apply(
                false, false, m, m, m, Hm, m, Hm, m, M, m, mult_ws, mult_ws_size);
            for (linalg_int i = 0; i < m; ++i)
            {
                for (linalg_int j = 0; j < m; ++j)
                {
                    M[i * m + j] += ((i == j) ? det : T(0)) - tr * Hm[i * m + j];
                }
            }
        }

        // Q, R <- qr(M) via the same native Householder QR qr_decomposition.cpp
        // exposes, instead of a self-contained copy of that algorithm.
        qr_decomposition<T, linalg_backend::native>::apply(
            m, m, M, m, Q, m, R, m, qr_ws, qr_ws_size);

        {
            LINALG_SCOPE("general_eigen/similarity");
            // Hm <- Q^T * Hm * Q, again via the shared native GEMM entry point.
            multiplication<T, linalg_backend::native>::apply(
                true, false, m, m, m, Q, m, Hm, m, tmp, m, mult_ws, mult_ws_size);
            // Product goes to the (now free) stride-m buffer M, then is copied into
            // H's stride-n leading block.
            multiplication<T, linalg_backend::native>::apply(
                false, false, m, m, m, tmp, m, Q, m, M, m, mult_ws, mult_ws_size);
            for (linalg_int i = 0; i < m; ++i)
            {
                for (linalg_int j = 0; j < m; ++j)
                {
                    H[static_cast<std::size_t>(i * n + j)] = M[i * m + j];
                }
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
    linalg_int*                   pivot,
    T*                            x)
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

    // M is the same matrix across every iteration below, so factor it once via
    // the shared native LU entry point lu_decomposition.cpp exposes, instead of
    // a self-contained Gaussian elimination redone from scratch each time.
    if (!lu_decomposition<T, linalg_backend::native>::apply(M, n, pivot, nullptr, 0))
    {
        return false;
    }

    bool ok = true;
    for (int it = 0; it < 25 && ok; ++it)
    {
        // Solve M * x = x in place, reusing the LU factors above, via the
        // shared native linear_solver entry point.
        linear_solver<T, linalg_backend::native>::apply(
            M, pivot, n, x, linear_solver_type::LU_UPFRONT_LINEAR_SOLVER, nullptr, 0);

        T norm = T(0);
        for (linalg_int i = 0; i < n; ++i)
        {
            norm += x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
        }
        norm = std::sqrt(norm);
        if (norm > T(0))
        {
            for (linalg_int i = 0; i < n; ++i)
            {
                x[static_cast<std::size_t>(i)] /= norm;
            }
        }
        else
        {
            ok = false;
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
    LINALG_SCOPE("eigen/native<float>");
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
    LINALG_SCOPE("eigen/native<double>");
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
            bufs.qr_ws,
            bufs.mult_ws))
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
        if (!inverse_iteration<T>(
                n, A, lda, eigenvalues_real[i], bufs.evec, bufs.inv_M, bufs.pivot, bufs.x))
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
    LINALG_SCOPE("general_eigen/native<float>");
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
    LINALG_SCOPE("general_eigen/native<double>");
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
