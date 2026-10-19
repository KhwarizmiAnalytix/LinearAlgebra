#include "include/matrix_operation/qr_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/workspace_arena.h"

#include <algorithm>
#include <cmath>

#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

#include "include/common/linalg_profiler.h"

namespace linalg
{

// Null/positivity checks shared by every qr_decomposition<T, Backend>::apply
// below, so callers (tests included) get the same validation regardless of
// which backend is active. There is no non-detail qr_decomposition()
// wrapper to put this in any more — callers use
// qr_decomposition<T, Backend>::apply directly, naming Backend explicitly.
static void check_qr_inputs(const void* A,
    const void*                         Q,
    const void*                         R,
    linalg_long                         rows,
    linalg_long                         columns,
    linalg_long                         lda,
    linalg_long                         ldq,
    linalg_long                         ldr)
{
    LINALG_CHECK(A != nullptr && Q != nullptr && R != nullptr,
        "qr_decomposition::apply: A, Q, and R must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldq != 0 && ldr != 0,
        "qr_decomposition::apply: rows/columns/lda/ldq/ldr must be positive");
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/qr_decomposition_mkl.cpp and
// blas/qr_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

// y[0..len) += alpha * x[0..len), both contiguous -- the row-axpy this native QR is
// built on (see qr_householder's comment below). Mirrors cholesky_decomposition.cpp's
// cholesky_dot: same linalg::packet<T> (src/common/packet.h) SIMD abstraction, same
// scalar-remainder tail, just an axpy instead of a dot product.
template <typename T> inline void qr_axpy(T* y, const T* x, T alpha, linalg_long len)
{
#if LINALG_HAS_VECTORIZATION
    using P                     = packet<T>;
    constexpr linalg_long width = P::size;

    typename P::packet_t pa;
    P::set(alpha, pa);

    linalg_long c = 0;
    for (; c + width <= len; c += width)
    {
        typename P::packet_t px;
        typename P::packet_t py;
        P::loadu(x + c, px);
        P::loadu(y + c, py);
        P::fma(pa, px, py, py);
        P::storeu(py, y + c);
    }
    for (; c < len; ++c)
    {
        y[c] += alpha * x[c];
    }
#else
    for (linalg_long c = 0; c < len; ++c)
    {
        y[c] += alpha * x[c];
    }
#endif
}

template <typename T> struct qr_householder_layout
{
    T* Rwork;
    T* Qwork;
    T* diag;
    T* tau;
    T* w;
};

template <typename T>
qr_householder_layout<T> qr_householder_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns)
{
    const linalg_long k = std::min(rows, columns);

    qr_householder_layout<T> p;
    p.Rwork = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.Qwork = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
    p.diag  = arena.next<T>(static_cast<size_t>(k));
    p.tau   = arena.next<T>(static_cast<size_t>(k));
    p.w     = arena.next<T>(static_cast<size_t>(std::max(columns, k)));
    return p;
}

template <typename T>
size_t qr_householder_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    qr_householder_plan<T>(arena, rows, columns);
    return arena.total_size();
}

// Unblocked Householder QR, following the same decomposition Eigen's HouseholderQR
// (Eigen/src/QR/HouseholderQR.h) and LAPACK's dgeqrf/dorgqr use, adapted to this
// library's caller-owned-workspace convention and row-major storage:
//
//  - Each reflector H_i = I - tau_i * v_i * v_i^T is applied to the trailing columns
//    of R only (never recomputed for column i itself, whose post-reflection values --
//    zero below the diagonal, alpha on it -- are already known analytically). Column i
//    of Rwork therefore keeps the *unnormalized* essential Householder vector v_i after
//    step i runs, exactly the LAPACK/Eigen compact storage trick: no separate per-column
//    v buffer, and R's implicit zeros are reconstructed from `diag` on output instead of
//    ever being written into Rwork.
//  - Q is never accumulated during elimination. Instead every reflector's (v_i, tau_i)
//    is kept (v_i in Rwork's column i as above, tau_i in `tau`) and Q is formed once,
//    economy-size (rows x k) directly, by applying H_{k-1}..H_0 in reverse to the first
//    k columns of the identity -- the same thing Eigen's HouseholderSequence does when
//    asked for a thin Q. For tall-skinny input (rows >> columns) this is the real win
//    over accumulating a full rows x rows Q one reflector at a time: O(rows * k^2)
//    instead of O(rows^2 * k).
//  - Each reflector application (to R's trailing columns, or to Q) is a rank-1 update
//    M -= tau * v * (v^T M) over a row-major submatrix. Computed as two passes of
//    contiguous row-axpy (qr_axpy above) -- accumulate w = v^T M by walking M one row at
//    a time (w += v[r] * M[r,:]), then apply M[r,:] -= (tau*v[r]) * w -- instead of one
//    dot product per output column, which would stride across rows in this row-major
///   layout. Same GEMV-then-GER shape Eigen's applyHouseholderOnTheLeft uses; vectorized
//    here via linalg::packet<T> the same way multiplication.cpp's native GEMM and
//    cholesky_decomposition.cpp's blocked kernel are.
template <typename T>
void qr_householder(linalg_long rows,
    linalg_long                 columns,
    const T*                    A,
    linalg_long                 lda,
    T*                          Q,
    linalg_long                 ldq,
    T*                          R,
    linalg_long                 ldr,
    void*                       workspace,
    size_t                      workspace_size)
{
    const linalg_long k = std::min(rows, columns);

    LINALG_CHECK(workspace_size >= qr_householder_workspace_size_impl<T>(rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs  = qr_householder_plan<T>(arena, rows, columns);
    auto* const             Rwork = bufs.Rwork;
    auto* const             Qwork = bufs.Qwork;
    auto* const             diag  = bufs.diag;
    auto* const             tau   = bufs.tau;
    auto* const             w     = bufs.w;

    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            Rwork[i * columns + j] = A[i * lda + j];
        }
    }

    // Forward pass: reduce Rwork to upper-trapezoidal form, column by column, keeping
    // each reflector's essential vector in place (see function comment above).
    for (linalg_long i = 0; i < k; ++i)
    {
        T norm = T(0);
        for (linalg_long r = i; r < rows; ++r)
        {
            const T x = Rwork[r * columns + i];
            norm += x * x;
        }
        norm = std::sqrt(norm);

        const T diagval = Rwork[i * columns + i];
        if (norm == T(0))
        {
            diag[i] = diagval;
            tau[i]  = T(0);
            continue;
        }

        const T alpha = (diagval >= T(0)) ? -norm : norm;
        diag[i]       = alpha;

        // v_i overwrites column i in place: v_i[i] = diag - alpha, v_i[r] = A[r][i]
        // unchanged for r > i (both already sitting at Rwork[r*columns+i]).
        Rwork[i * columns + i] = diagval - alpha;

        T vTv = T(0);
        for (linalg_long r = i; r < rows; ++r)
        {
            const T vr = Rwork[r * columns + i];
            vTv += vr * vr;
        }
        if (vTv == T(0))
        {
            tau[i] = T(0);
            continue;
        }
        // LAPACK/Eigen convention: leave v unnormalized and fold the normalization
        // into a single scalar tau = 2 / (v^T v), so H = I - tau*v*v^T equals
        // I - 2*(v/||v||)*(v/||v||)^T without an extra O(rows) pass to compute ||v||
        // and divide by it.
        tau[i] = T(2) / vTv;

        const linalg_long trailing = columns - i - 1;
        if (trailing == 0)
        {
            continue;
        }

        // w = v_i^T * Rwork[i:rows, i+1:columns], accumulated one row at a time so each
        // contribution is a contiguous axpy over the trailing columns.
        std::fill_n(w, static_cast<size_t>(trailing), T(0));
        for (linalg_long r = i; r < rows; ++r)
        {
            const T vr = Rwork[r * columns + i];
            if (vr != T(0))
            {
                qr_axpy(w, &Rwork[r * columns + i + 1], vr, trailing);
            }
        }
        // Rwork[i:rows, i+1:columns] -= tau * v_i * w^T.
        for (linalg_long r = i; r < rows; ++r)
        {
            const T vr = Rwork[r * columns + i];
            if (vr == T(0))
            {
                continue;
            }
            qr_axpy(&Rwork[r * columns + i + 1], w, -tau[i] * vr, trailing);
        }
    }

    // Form the economy-size Q (rows x k) by applying H_{k-1} .. H_0 in reverse to the
    // first k columns of the identity -- see function comment above.
    std::fill_n(Qwork, static_cast<size_t>(rows) * static_cast<size_t>(k), T(0));
    for (linalg_long r = 0; r < k; ++r)
    {
        Qwork[r * k + r] = T(1);
    }

    // linalg_long is unsigned, so this walks the index down via an ascending counter
    // rather than comparing a decrementing `i` against zero (which would never be
    // false and wrap around instead).
    for (linalg_long step = 0; step < k; ++step)
    {
        const linalg_long i = k - 1 - step;
        if (tau[i] == T(0))
        {
            continue;
        }

        // w = v_i^T * Qwork[i:rows, 0:k].
        std::fill_n(w, static_cast<size_t>(k), T(0));
        for (linalg_long r = i; r < rows; ++r)
        {
            const T vr = Rwork[r * columns + i];
            if (vr != T(0))
            {
                qr_axpy(w, &Qwork[r * k], vr, k);
            }
        }
        // Qwork[i:rows, 0:k] -= tau * v_i * w^T.
        for (linalg_long r = i; r < rows; ++r)
        {
            const T vr = Rwork[r * columns + i];
            if (vr == T(0))
            {
                continue;
            }
            qr_axpy(&Qwork[r * k], w, -tau[i] * vr, k);
        }
    }

    for (linalg_long r = 0; r < rows; ++r)
    {
        for (linalg_long c = 0; c < k; ++c)
        {
            Q[r * ldq + c] = Qwork[r * k + c];
        }
    }

    for (linalg_long r = 0; r < k; ++r)
    {
        for (linalg_long c = 0; c < columns; ++c)
        {
            R[r * ldr + c] = (c == r) ? diag[r] : ((c > r) ? Rwork[r * columns + c] : T(0));
        }
    }
}

}  // namespace

template <>
void qr_decomposition<float, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                         columns,
    const float*                                                        A,
    linalg_long                                                         lda,
    float*                                                              Q,
    linalg_long                                                         ldq,
    float*                                                              R,
    linalg_long                                                         ldr,
    void*                                                               workspace,
    size_t                                                              workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/native<float>");
    check_qr_inputs(A, Q, R, rows, columns, lda, ldq, ldr);
    qr_householder(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
}

template <>
void qr_decomposition<double, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                          columns,
    const double*                                                        A,
    linalg_long                                                          lda,
    double*                                                              Q,
    linalg_long                                                          ldq,
    double*                                                              R,
    linalg_long                                                          ldr,
    void*                                                                workspace,
    size_t                                                               workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/native<double>");
    check_qr_inputs(A, Q, R, rows, columns, lda, ldq, ldr);
    qr_householder(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
}

template <>
size_t qr_decomposition<float, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return qr_householder_workspace_size_impl<float>(rows, columns);
}

template <>
size_t qr_decomposition<double, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return qr_householder_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg
