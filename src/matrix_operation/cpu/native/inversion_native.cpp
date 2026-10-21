#include "include/matrix_operation/inversion.h"

#include "include/common/linalg_checks.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"

#include <algorithm>

#include "include/common/linalg_macros.h"
#include "include/common/workspace_arena.h"

#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

#include "include/common/linalg_profiler.h"

namespace linalg
{

// Null checks shared by every inversion<T, Backend>::apply below, so
// callers (tests included) get the same validation regardless of which
// backend is active. There is no non-detail matrix_invert() wrapper to put
// this in any more — callers use inversion<T, linalg_backend::NATIVE>::apply
// directly.
static void check_invert_inputs(const void* m, linalg_int lda)
{
    LINALG_CHECK(m != nullptr, "inversion::apply: m must not be null");
    LINALG_CHECK(lda > 0, "inversion::apply: lda must be positive", lda);
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/matrix_inversion_mkl.cpp and
// blas/matrix_inversion_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

template <typename T> size_t invert_scalar_workspace_size(linalg_int lda, linear_solver_type type)
{
    detail::workspace_arena arena;
    const auto              n = static_cast<size_t>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        // One lda x lda scratch of the evolving inverse -- see lu_invert below.
        arena.next<T>(n * n);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        // col_scratch (one column of L^-1 being built) then xt (L^-1,
        // transposed) -- see cholesky_invert below.
        arena.next<T>(n);
        arena.next<T>(n * n);
        break;
    default:
        break;
    }
    return arena.total_size();
}

namespace
{

// y[0..len) -= a * x[0..len), vectorized via linalg::packet<T> (src/common/
// packet.h) -- the same Eigen-derived SIMD abstraction, and the same kernel
// shape, as lu_decomposition_native.cpp's lu_axpy_sub (this file keeps its
// own internal-linkage copy, same reasoning as check_invert_inputs above:
// each backend/kernel translation unit keeps its own so linking several
// cannot clash). Used below to turn lu_invert's forward/back substitution
// into full-row updates across every column of the inverse at once, instead
// of one O(i)-length dot product per column.
// y and x are __restrict: every call site below passes two different rows
// of the same matrix (forward substitution has k < i, back substitution
// has k > i), so they never alias -- unlike invert_dot below, which is
// called with x == y at the diagonal and must not be.
template <typename T> inline void inv_axpy_sub(T* __restrict y, const T* __restrict x, T a, linalg_int len)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;

    typename P::packet_t va;
    P::set(-a, va);

    linalg_int c = 0;
    for (; c + 4 * width <= len; c += 4 * width)
    {
        typename P::packet_t vx0, vx1, vx2, vx3;
        typename P::packet_t vy0, vy1, vy2, vy3;
        typename P::packet_t vo0, vo1, vo2, vo3;
        P::loadu(x + c, vx0);
        P::loadu(x + c + width, vx1);
        P::loadu(x + c + 2 * width, vx2);
        P::loadu(x + c + 3 * width, vx3);
        P::loadu(y + c, vy0);
        P::loadu(y + c + width, vy1);
        P::loadu(y + c + 2 * width, vy2);
        P::loadu(y + c + 3 * width, vy3);
        P::fma(va, vx0, vy0, vo0);
        P::fma(va, vx1, vy1, vo1);
        P::fma(va, vx2, vy2, vo2);
        P::fma(va, vx3, vy3, vo3);
        P::storeu(vo0, y + c);
        P::storeu(vo1, y + c + width);
        P::storeu(vo2, y + c + 2 * width);
        P::storeu(vo3, y + c + 3 * width);
    }
    for (; c + width <= len; c += width)
    {
        typename P::packet_t vx;
        typename P::packet_t vy;
        typename P::packet_t vout;
        P::loadu(x + c, vx);
        P::loadu(y + c, vy);
        P::fma(va, vx, vy, vout);
        P::storeu(vout, y + c);
    }
    for (; c < len; ++c)
    {
        y[c] -= a * x[c];
    }
#else
    for (linalg_int c = 0; c < len; ++c)
    {
        y[c] -= a * x[c];
    }
#endif
}

// Dot product of two length-`len` contiguous operands, vectorized the same
// way cholesky_decomposition_native.cpp's cholesky_dot and
// lu_decomposition_native.cpp's lu_dot are (same reduction shape, own
// internal-linkage copy per the convention noted above). Used below by
// cholesky_invert once its two reductions are rewritten to read contiguous
// scratch instead of a strided matrix column.
//
// x and y are deliberately NOT __restrict: cholesky_invert's pass C calls
// this with x == y at the diagonal (i == j), computing a self dot product
// (sum of squares) -- marking both restrict would assert they never alias
// and then immediately violate that assertion at a real call site.
template <typename T> inline T invert_dot(const T* x, const T* y, linalg_int len)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;

    typename P::packet_t acc0, acc1, acc2, acc3;
    P::set(static_cast<T>(0), acc0);
    P::set(static_cast<T>(0), acc1);
    P::set(static_cast<T>(0), acc2);
    P::set(static_cast<T>(0), acc3);

    linalg_int c = 0;
    for (; c + 4 * width <= len; c += 4 * width)
    {
        typename P::packet_t px0, px1, px2, px3;
        typename P::packet_t py0, py1, py2, py3;
        P::loadu(x + c, px0);
        P::loadu(x + c + width, px1);
        P::loadu(x + c + 2 * width, px2);
        P::loadu(x + c + 3 * width, px3);
        P::loadu(y + c, py0);
        P::loadu(y + c + width, py1);
        P::loadu(y + c + 2 * width, py2);
        P::loadu(y + c + 3 * width, py3);
        P::fma(px0, py0, acc0, acc0);
        P::fma(px1, py1, acc1, acc1);
        P::fma(px2, py2, acc2, acc2);
        P::fma(px3, py3, acc3, acc3);
    }

    typename P::packet_t acc01, acc23, acc;
    P::add(acc0, acc1, acc01);
    P::add(acc2, acc3, acc23);
    P::add(acc01, acc23, acc);

    for (; c + width <= len; c += width)
    {
        typename P::packet_t px;
        typename P::packet_t py;
        P::loadu(x + c, px);
        P::loadu(y + c, py);
        P::fma(px, py, acc, acc);
    }

    T sum = P::predux(acc);
    for (; c < len; ++c)
    {
        sum += x[c] * y[c];
    }
    return sum;
#else
    T sum = static_cast<T>(0);
    for (linalg_int c = 0; c < len; ++c)
    {
        sum += x[c] * y[c];
    }
    return sum;
#endif
}

// Inverts an LU-factored matrix by solving L U X = P (P the row-permuted
// identity) for all n columns of X at once -- replacing the old per-column
// forward/back substitution, whose reduction length grew/shrank with the
// row index and so only ever vectorized a single narrow dot product at a
// time. Every inner update here is instead a full-width, length-lda AXPY
// across an entire row of the evolving inverse (inv_axpy_sub above),
// embarrassingly vectorizable and reusing the exact kernel shape
// lu_decomposition_native.cpp's own blocked panel solve is built on.
//
// inv_m is a caller-owned lda x lda scratch buffer of element type T. The
// previous version kept this scratch in double for both float and double,
// but that only ever protected the back-substitution half: the
// forward-substitution accumulator there was already typed T (narrowing
// back every step), so the "double precision" guarantee was already only
// half real. This rewrite drops it along with the vectorization, rather
// than keeping a cost (double the scratch for float) that wasn't actually
// buying full double precision.
// m and inv_m are __restrict: inv_m is the caller's workspace-arena scratch
// (inversion.h documents it as a separate caller-owned buffer from m), so
// the two never overlap. pivot is a distinct, separately-owned array too.
template <typename T>
void lu_invert(T* __restrict m, LINALG_UNUSED const linalg_int* __restrict pivot, linalg_int lda,
    T* __restrict inv_m)
{
    using size_type = linalg_int;

    for (size_type i = 0; i < lda; ++i)
    {
        T* const __restrict row = inv_m + i * lda;
        std::fill_n(row, lda, static_cast<T>(0));
        row[i] = static_cast<T>(1);
    }

#if LINALG_HAS_LU_PIVOTING
    // The same row-interchange sequence linear_solver_native.cpp's lu_solve
    // applies to a single right-hand side, applied here to every column at
    // once via a full-row swap: mathematically identical to swapping each
    // column's two entries independently, since the sequence depends only
    // on pivot[], not on which column is being solved. row_i/row_p are safe
    // to mark __restrict relative to each other: this branch only runs
    // when p != i.
    for (size_type i = 0; i < lda; ++i)
    {
        const size_type p = pivot[i] - 1;
        if (p != i)
        {
            T* const __restrict row_i = inv_m + i * lda;
            T* const __restrict row_p = inv_m + p * lda;
            std::swap_ranges(row_i, row_i + lda, row_p);
        }
    }
#endif

    // Forward substitution against the unit-lower triangle (L's
    // multipliers, stored strictly below the diagonal of m). m_row_i is
    // hoisted once per i and indexed by k, rather than m's row base being
    // recomputed from scratch on every k -- the row base never changes
    // within this inner loop. row_i and inv_m + k * lda are always
    // different rows (k < i here), so inv_axpy_sub's __restrict contract
    // holds.
    for (size_type i = 0; i < lda; ++i)
    {
        T* const __restrict       row_i   = inv_m + i * lda;
        const T* const __restrict m_row_i = m + i * lda;
        for (size_type k = 0; k < i; ++k)
        {
            inv_axpy_sub(row_i, inv_m + k * lda, m_row_i[k], lda);
        }
    }

    // Back substitution against the upper triangle (including the
    // diagonal; k > i here, so again never the same row as row_i). The
    // final per-row scale by 1/m_row_i[i] is left scalar: it's O(lda) per
    // row, O(lda^2) total, dominated by the O(lda^3) AXPY passes above, so
    // vectorizing it would add complexity without a measurable payoff.
    for (size_type i = lda - 1; i >= 0; --i)
    {
        T* const __restrict       row_i   = inv_m + i * lda;
        const T* const __restrict m_row_i = m + i * lda;
        for (size_type k = i + 1; k < lda; ++k)
        {
            inv_axpy_sub(row_i, inv_m + k * lda, m_row_i[k], lda);
        }
        const T inv_diag = static_cast<T>(1) / m_row_i[i];
        for (size_type c = 0; c < lda; ++c)
        {
            row_i[c] *= inv_diag;
        }
    }

    for (size_type i = 0; i < lda; ++i)
    {
        std::copy_n(inv_m + i * lda, lda, m + i * lda);
    }
}

// Inverts a lower-Cholesky-factored matrix (A = L L^T, L in the lower
// triangle of m; the upper triangle is zero per
// cholesky_decomposition::apply's contract) in three passes, replacing the
// old interleaved column-by-column recurrence whose two reductions read a
// strided matrix column (m[k * lda + j], stride lda) and so could not be
// vectorized against a contiguous row without a per-element gather
// (packet<T>::gather is itself a scalar loop -- see src/common/packet.h --
// so routing through it would add indirection without adding real SIMD
// throughput):
//
//   A. Invert L in place (X = L^-1, lower triangular, overwriting L) one
//      column at a time with the same recurrence as before, but keeping
//      the column just computed in a contiguous col_scratch buffer so the
//      reduction becomes a plain row-vs-buffer dot product instead of a
//      row-vs-column one.
//   B. Transpose X into a contiguous scratch xt (xt[p*lda+q] = X(q, p),
//      only ever written/read for q >= p -- X's structural zero region
//      above the diagonal is never touched).
//   C. A^-1(i, j) for i <= j (the upper triangle, mirrored to the lower
//      triangle at the end exactly as before) is dot(xt row i, xt row j)
//      starting at column j -- two contiguous rows, fully vectorizable via
//      invert_dot. The old code computed this same reduction for every row
//      i in [0, lda) and only ever kept the i <= j results, so this also
//      roughly halves that pass's work by not computing i > j at all.
//
// col_scratch is lda elements; xt is lda*lda -- see invert_scalar_workspace_size.
// m, col_scratch, and xt are __restrict: col_scratch and xt are two
// disjoint sub-ranges carved from the caller's workspace by
// workspace_arena (see matrix_invert_scalar_impl below), and that
// workspace is itself documented as a separate caller-owned buffer from m.
template <typename T>
void cholesky_invert(T* __restrict m, linalg_int lda, T* __restrict col_scratch, T* __restrict xt)
{
    using size_type = linalg_int;

    // A. X = L^-1, in place, lower triangle of m. row_i is hoisted once per
    // i (it's read at column j and written at column j, both against the
    // same base) instead of m's row base being recomputed for each access.
    // row_j and row_i are safe to mark __restrict relative to each other:
    // i > j always in the inner loop below.
    for (size_type j = 0; j < lda; ++j)
    {
        T* const __restrict row_j = m + j * lda;
        const T             diag  = static_cast<T>(1) / row_j[j];
        row_j[j]                  = diag;
        col_scratch[j]            = diag;

        for (size_type i = j + 1; i < lda; ++i)
        {
            T* const __restrict row_i = m + i * lda;
            const T val = -invert_dot(row_i + j, col_scratch + j, i - j) / row_i[i];
            row_i[j]       = val;
            col_scratch[i] = val;
        }
    }

    // B. xt = X^T (upper-triangular pattern: xt[p][q] is only ever written,
    // or read by pass C below, for q >= p). xt_row is hoisted once per p
    // (reused for every q), and col_ptr walks column p of m one row at a
    // time via a +lda pointer step instead of recomputing q*lda+p for every
    // q. xt_row (from xt) and col_ptr (from m) are different buffers.
    for (size_type p = 0; p < lda; ++p)
    {
        T* const __restrict xt_row  = xt + p * lda;
        const T* __restrict col_ptr = m + p * lda + p;
        for (size_type q = p; q < lda; ++q, col_ptr += lda)
        {
            xt_row[q] = *col_ptr;
        }
    }

    // C. A^-1, upper triangle (i <= j), via contiguous row dot products.
    // xt_row_j is fixed for the whole i loop below (every i dots against
    // the same row j of xt), so it's hoisted out instead of being
    // recomputed from j*lda+j on every i. Not passed through a __restrict
    // parameter here: invert_dot deliberately isn't __restrict-qualified
    // (see its definition above) since i == j makes this a self dot
    // product.
    for (size_type j = 0; j < lda; ++j)
    {
        const T* const xt_row_j = xt + j * lda + j;
        for (size_type i = 0; i <= j; ++i)
        {
            m[i * lda + j] = invert_dot(xt + i * lda + j, xt_row_j, lda - j);
        }
    }

    // Mirror the upper triangle into the lower one. row_i is hoisted once
    // per i (reused for every j), and col_ptr walks column i one row at a
    // time via a +lda pointer step instead of recomputing j*lda+i for
    // every j. row_i (row i, columns < i) and col_ptr (column i, rows < i)
    // are safe to mark __restrict relative to each other: row_i's row
    // index is always i, col_ptr's row index is always < i, so they can
    // never address the same element.
    for (size_type i = 1; i < lda; ++i)
    {
        T* const __restrict row_i   = m + i * lda;
        const T* __restrict col_ptr = m + i;
        for (size_type j = 0; j < i; ++j, col_ptr += lda)
        {
            row_i[j] = *col_ptr;
        }
    }
}

template <typename T>
void matrix_invert_scalar_impl(T* __restrict m,
    linalg_int* __restrict        pivot,
    linalg_int                    lda,
    linear_solver_type            type,
    void*                         workspace,
    size_t                        workspace_size)
{
    LINALG_CHECK(workspace_size >= invert_scalar_workspace_size<T>(lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<size_t>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        auto* inv_m = arena.next<T>(n * n);
        if (type == linear_solver_type::LU_UPFRONT_LINEAR_SOLVER ||
            lu_decomposition<T, linalg_backend::NATIVE>::apply(m, lda, pivot, nullptr, 0))
        {
            lu_invert(m, pivot, lda, inv_m);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
    {
        auto* col_scratch = arena.next<T>(n);
        auto* xt           = arena.next<T>(n * n);
        if (type == linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER ||
            cholesky_decomposition<T, linalg_backend::NATIVE>::apply(
                m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR, nullptr, 0, nullptr))
        {
            cholesky_invert(m, lda, col_scratch, xt);
        }
        break;
    }
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

}  // namespace

template <>
void inversion<float, linalg_backend::NATIVE>::apply(float* __restrict m,
    linalg_int* __restrict                                  pivot,
    linalg_int                                              lda,
    linear_solver_type                                      type,
    void*                                                   workspace,
    size_t                                                  workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("inversion/native<float>");
    check_invert_inputs(m, lda);
    matrix_invert_scalar_impl(m, pivot, lda, type, workspace, workspace_size);
}

template <>
void inversion<double, linalg_backend::NATIVE>::apply(double* __restrict m,
    linalg_int* __restrict                                    pivot,
    linalg_int                                                lda,
    linear_solver_type                                        type,
    void*                                                     workspace,
    size_t                                                    workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("inversion/native<double>");
    check_invert_inputs(m, lda);
    matrix_invert_scalar_impl(m, pivot, lda, type, workspace, workspace_size);
}

template <>
size_t inversion<float, linalg_backend::NATIVE>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return invert_scalar_workspace_size<float>(lda, type);
}

template <>
size_t inversion<double, linalg_backend::NATIVE>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return invert_scalar_workspace_size<double>(lda, type);
}

}  // namespace linalg
