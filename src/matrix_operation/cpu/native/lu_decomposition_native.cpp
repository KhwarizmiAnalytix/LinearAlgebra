#include "include/matrix_operation/lu_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/backend.h"
#include "include/common/workspace_arena.h"
#include "include/matrix_operation/multiplication.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "include/common/linalg_macros.h"

#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

#include "include/common/linalg_profiler.h"

namespace linalg
{

// Null/positivity checks shared by every lu_decomposition<T, Backend>::apply
// below, so callers (linear_solver.cpp, inversion.cpp, tests) get the
// same validation regardless of which backend is active. There is no
// non-detail lu_decomposition() wrapper to put this in any more — callers
// use lu_decomposition<T, Backend>::apply directly, naming Backend explicitly.
static void check_lu_inputs(const void* m, const void* pivot, linalg_int lda)
{
    LINALG_CHECK(
        m != nullptr && pivot != nullptr, "lu_decomposition::apply: m and pivot must not be null");
    LINALG_CHECK(lda > 0, "lu_decomposition::apply: lda must be positive", lda);
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/lu_decomposition_mkl.cpp and
// blas/lu_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

// y[c] -= a * x[c] for c in [0, len), the axpy this blocked LU's panel
// elimination, triangular solve, and rank-bs trailing update all reduce
// to (see lu_decomposition_blocked_impl below). Vectorized via
// linalg::packet<T> (src/common/packet.h) -- the same Eigen-derived SIMD
// abstraction cholesky_decomposition.cpp's blocked kernel and
// multiplication.cpp's native GEMM micro-kernel are built on -- instead of
// relying on the compiler to auto-vectorize a scalar loop with a
// runtime-variable trip count.
//
// Processes 4*width elements per iteration across 4 independent packet_t
// slots rather than one width-at-a-time. Each output element is already
// independent of every other (no cross-iteration dependency, unlike
// lu_dot's reduction below), so correctness doesn't need the unrolling --
// but a single-slot version benchmarked ~2x slower than the plain scalar
// loop it replaced: that loop's trip count is small enough per call (it
// shrinks every panel column, see lu_panel_factor) that clang's own
// auto-vectorizer, left to vectorize the scalar form itself, chose to
// unroll 4x with 4 independent accumulators precisely to amortize
// per-iteration loop overhead (compare/increment/branch) over more work;
// a hand-vectorized loop that always does only one packet_t per iteration
// gives that back. Matching that unroll factor here recovers it.
// y and x are __restrict: every call site below passes two different rows
// of m (lu_panel_factor's r > i, the triangular solve's p < ii), so they
// never alias.
template <typename T>
inline void lu_axpy_sub(T* __restrict y, const T* __restrict x, T a, linalg_long len)
{
#if LINALG_HAS_VECTORIZATION
    using P                     = packet<T>;
    constexpr linalg_long width = P::size;

    typename P::packet_t va;
    P::set(-a, va);

    linalg_long c = 0;
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
    for (linalg_long c = 0; c < len; ++c)
    {
        y[c] -= a * x[c];
    }
#endif
}

// Unblocked Gaussian elimination with (optional) partial pivoting, exactly
// the algorithm this file used before blocking: single source of truth for
// both the scalar path used below kLuBlockSize and, blocked into panels of
// width kLuBlockSize, the right-looking blocked kernel below it. Mirrors
// LAPACK's dgetrf2 unblocked base case and Eigen's PartialPivLU unblocked
// path (Eigen/src/LU/PartialPivLU.h, unblocked_lu) the same way; `m`/`lda`
// address the full matrix, but only the `width`-wide column range starting
// at `(start, start)` is factored -- rows below `start` through `lda - 1`
// participate in the (row-only, i.e. partial, never column) pivot search,
// but the swap once a pivot row is found only touches columns
// `[0, start + width)` -- the columns already factored (previous panels)
// plus this panel's own width -- instead of the entire row out to
// `lda - 1`. Columns at or past `start + width` haven't been touched by
// this panel yet and are deliberately left unswapped here: the accumulated
// `pivot` sequence this leaves behind gets applied to them in one batched
// pass by lu_apply_pivots_to_trailing below, exactly the split LAPACK's
// blocked dgetrf2 makes between its recursive panel factorization and the
// dlaswp call that follows it -- eagerly swapping the full row on every
// pivot found would touch trailing memory this panel has no other reason
// to touch yet, once per row per panel column.
// m and pivot are __restrict: they're always two distinct caller-owned
// buffers (lu_decomposition.h documents pivot as a separate output array).
template <typename T>
bool lu_panel_factor(T* __restrict m,
    linalg_long lda,
    linalg_long start,
    linalg_long width,
    LINALG_UNUSED linalg_int* __restrict pivot,
    LINALG_UNUSED T tolerance)
{
    using size_type = linalg_long;

    for (size_type ii = 0; ii < width; ++ii)
    {
        const size_type i = start + ii;
        T* const        row_i_f =
            m + i * lda;  // row i, full width -- read/written at several columns below
#if LINALG_HAS_LU_PIVOTING
        T         maxA    = std::abs(row_i_f[i]);
        size_type imax    = i;
        const T*  col_ptr = m + (i + 1) * lda + i;  // walks column i, one row at a time
        for (size_type k = i + 1; k < lda; ++k, col_ptr += lda)
        {
            const auto absA = std::fabs(*col_ptr);
            if (absA > maxA)
            {
                maxA = absA;
                imax = k;
            }
        }
        if (maxA < tolerance)
        {
            return false;
        }
        if (imax != i)
        {
            pivot[i] = pivot[imax];
            // row_i_f and row_imax_f are safe to treat as non-aliasing
            // here: this branch only runs when imax != i.
            T* const __restrict row_imax_f = m + imax * lda;
            for (size_type j = 0; j < start + width; ++j)
            {
                std::swap(row_i_f[j], row_imax_f[j]);
            }
        }
#endif
        // row_i_f[i] re-reads live memory (not a cached value), so this
        // correctly sees the post-swap diagonal when a swap happened above.
        const auto norm         = static_cast<T>(1. / row_i_f[i]);
        T* const __restrict l_i = row_i_f + start;

        // l_i (row i) and l_r (row r, r > i always) never alias, so
        // lu_axpy_sub's __restrict contract holds.
        for (size_type r = i + 1; r < lda; ++r)
        {
            T* const __restrict l_r = m + r * lda + start;
            l_r[ii] *= norm;
            lu_axpy_sub(l_r + ii + 1, l_i + ii + 1, l_r[ii], width - ii - 1);
        }
    }
    return true;
}

#if LINALG_HAS_LU_PIVOTING
// Applies the row interchanges lu_panel_factor recorded in pivot[start ..
// start+width) -- but deferred swapping on -- to the trailing columns
// [start+width, lda), in the same increasing-i order the panel's own
// swaps happened in. Same role as LAPACK's dlaswp call right after
// dgetrf2's recursive panel factorization: row i can only ever be chosen
// as a *target* (pivot[i']'s value) by a later step i' > i, never a
// *source* again once step i has run (the search at step i' only ranges
// over k in [i'+1, lda)), so replaying these swaps in order against any
// column range -- here the untouched trailing one -- reproduces exactly
// the permutation lu_panel_factor already applied to columns
// [0, start+width).
// m and pivot are __restrict, same reasoning as lu_panel_factor above.
template <typename T>
void lu_apply_pivots_to_trailing(T* __restrict m,
    linalg_long lda,
    linalg_long start,
    linalg_long width,
    const linalg_int* __restrict pivot)
{
    using size_type = linalg_long;

    for (size_type i = start; i < start + width; ++i)
    {
        const size_type piv = static_cast<size_type>(pivot[i]) - 1;
        if (piv != i)
        {
            // row_i and row_piv are safe to treat as non-aliasing: this
            // branch only runs when piv != i.
            T* const __restrict row_i   = m + i * lda;
            T* const __restrict row_piv = m + piv * lda;
            for (size_type j = start + width; j < lda; ++j)
            {
                std::swap(row_i[j], row_piv[j]);
            }
        }
    }
}
#endif

// Right-looking blocked LU with partial pivoting: the same shape PyTorch's
// CPU path gets for free by calling LAPACK's (internally blocked) getrf,
// and that Eigen's PartialPivLU (Eigen/src/LU/PartialPivLU.h) implements
// directly -- factor a kLuBlockSize-wide panel with the unblocked kernel
// above (which already applies that panel's row swaps to the whole row,
// so no deferred pivot application step is needed here), solve the panel's
// trailing row-block U12 = L11^-1 * A12 by forward substitution against
// L11, then fold L21 * U12 into the trailing submatrix with a real GEMM
// call (see lu_trailing_update_workspace_size/the trailing-update block
// below) before moving to the next panel. Degenerates to exactly the
// unblocked kernel above when lda <= kLuBlockSize.
inline constexpr linalg_long kLuBlockSize = 64;

// Per-panel workspace the trailing update below needs: a dense rest x rest
// scratch for L21 * U12's result (delta), plus whatever
// multiplication<T, NATIVE> itself needs to pack L21/U12 for its blocked
// GEBP kernel. Mirrors the arena-carve order the trailing update uses,
// exactly as workspace_arena's contract requires (see
// include/common/workspace_arena.h) so the two can never drift apart.
template <typename T>
size_t lu_trailing_update_workspace_size(linalg_long lda, linalg_long k, linalg_long bs)
{
    const linalg_long rest = lda - k - bs;
    if (rest <= 0)
    {
        return 0;
    }
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(rest) * static_cast<size_t>(rest));
    const size_t mm_bytes = multiplication<T, linalg_backend::NATIVE>::workspace_size(
        static_cast<linalg_int>(rest), static_cast<linalg_int>(rest), static_cast<linalg_int>(bs));
    arena.next<unsigned char>(mm_bytes);
    return arena.total_size();
}

// Walks the same k-loop lu_decomposition_blocked_impl below runs, without
// touching any matrix data, to find the true largest per-panel workspace
// requirement for this lda. rest shrinks and bs stays at kLuBlockSize (bar
// a possibly-smaller final panel, where rest is correspondingly small) as
// k increases, so the first panel is the practical worst case -- but this
// takes the actual max rather than assuming that, since
// multiplication<T, NATIVE>::workspace_size's cache-blocking math isn't
// provably monotonic in its inputs.
template <typename T> size_t lu_workspace_size(linalg_long lda)
{
    size_t max_bytes = 0;
    for (linalg_long k = 0; k < lda; k += kLuBlockSize)
    {
        const linalg_long bs = std::min(kLuBlockSize, lda - k);
        max_bytes = std::max(max_bytes, lu_trailing_update_workspace_size<T>(lda, k, bs));
    }
    return max_bytes;
}

// m and pivot are __restrict, same reasoning as lu_panel_factor above.
template <typename T>
bool lu_decomposition_blocked_impl(T* __restrict m,
    linalg_long lda,
    linalg_int* __restrict pivot,
    T      tolerance,
    void*  workspace,
    size_t workspace_size)
{
    using size_type = linalg_long;

#if LINALG_HAS_LU_PIVOTING
    for (size_type i = 0; i < lda; ++i)
    {
        pivot[i] = static_cast<linalg_int>(i + 1);
    }
#endif

    for (size_type k = 0; k < lda; k += kLuBlockSize)
    {
        const size_type bs = std::min(kLuBlockSize, lda - k);

#if LINALG_HAS_LU_PIVOTING
        if (!lu_panel_factor(m, lda, k, bs, pivot, tolerance))
        {
            return false;
        }
#else
        lu_panel_factor(m, lda, k, bs, pivot, tolerance);
#endif

        const size_type rest = lda - k - bs;
        if (rest == 0)
        {
            continue;
        }

#if LINALG_HAS_LU_PIVOTING
        lu_apply_pivots_to_trailing(m, lda, k, bs, pivot);
#endif

        // Triangular solve: U12 = L11^-1 * A12, one panel row at a time via
        // forward substitution against the unit-lower-triangular L11
        // already factored in place above (L11's multipliers live strictly
        // below the panel diagonal).
        // row_i (row k+ii) and row_p (row k+p, p < ii always) never alias,
        // so lu_axpy_sub's __restrict contract holds.
        for (size_type ii = 0; ii < bs; ++ii)
        {
            T* const row_full         = m + (k + ii) * lda;  // row k+ii, full width
            T* const __restrict row_i = row_full + k + bs;
            for (size_type p = 0; p < ii; ++p)
            {
                const T l_ip              = row_full[k + p];
                T* const __restrict row_p = m + (k + p) * lda + k + bs;
                lu_axpy_sub(row_i, row_p, l_ip, rest);
            }
        }

        // Trailing update: A22 -= L21 * U12. L21 (m + (k+bs)*lda + k, rest
        // rows x bs columns) and U12 (m + k*lda + (k+bs), bs rows x rest
        // columns) are both submatrices of m with its full row stride lda,
        // which multiplication<T, NATIVE>::apply accepts directly via its
        // own lda/ldb parameters -- no packing of either operand by this
        // file is needed. This is the real Level-3 GEMM call LAPACK's
        // dgetrf and Eigen's PartialPivLU defer this update to; delta
        // receives L21 * U12 (multiplication always replaces its output,
        // never accumulates), and the actual "-=" is a second pass reusing
        // lu_axpy_sub above with a scalar multiplier of 1.
        detail::workspace_arena arena(workspace, workspace_size);
        T* const     delta = arena.next<T>(static_cast<size_t>(rest) * static_cast<size_t>(rest));
        const size_t mm_bytes =
            multiplication<T, linalg_backend::NATIVE>::workspace_size(static_cast<linalg_int>(rest),
                static_cast<linalg_int>(rest),
                static_cast<linalg_int>(bs));
        unsigned char* const mm_workspace = arena.next<unsigned char>(mm_bytes);

        multiplication<T, linalg_backend::NATIVE>::apply(false,
            false,
            static_cast<linalg_int>(rest),
            static_cast<linalg_int>(rest),
            static_cast<linalg_int>(bs),
            m + (k + bs) * lda + k,
            static_cast<linalg_int>(lda),
            m + k * lda + (k + bs),
            static_cast<linalg_int>(lda),
            delta,
            static_cast<linalg_int>(rest),
            mm_workspace,
            mm_bytes);

        // out_row (columns [k+bs, lda)) and delta_row (a row of the
        // dense, freshly-written rest x rest scratch above) never alias,
        // so lu_axpy_sub's __restrict contract holds.
        for (size_type r = 0; r < rest; ++r)
        {
            T* const       out_row   = m + (k + bs + r) * lda + (k + bs);
            const T* const delta_row = delta + r * rest;
            lu_axpy_sub(out_row, delta_row, static_cast<T>(1), rest);
        }
    }
    return true;
}

}  // namespace

template <>
bool lu_decomposition<float, linalg_backend::NATIVE>::apply(float* __restrict m,
    linalg_int lda,
    linalg_int* __restrict pivot,
    void*  workspace,
    size_t workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("lu_decomposition/native<float>");
    check_lu_inputs(m, pivot, lda);
    return lu_decomposition_blocked_impl(
        m, lda, pivot, std::numeric_limits<float>::epsilon(), workspace, workspace_size);
}

template <>
bool lu_decomposition<double, linalg_backend::NATIVE>::apply(double* __restrict m,
    linalg_int lda,
    linalg_int* __restrict pivot,
    void*  workspace,
    size_t workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("lu_decomposition/native<double>");
    check_lu_inputs(m, pivot, lda);
    return lu_decomposition_blocked_impl(
        m, lda, pivot, std::numeric_limits<double>::epsilon(), workspace, workspace_size);
}

template <> size_t lu_decomposition<float, linalg_backend::NATIVE>::workspace_size(linalg_int lda)
{
    return lu_workspace_size<float>(lda);
}

template <> size_t lu_decomposition<double, linalg_backend::NATIVE>::workspace_size(linalg_int lda)
{
    return lu_workspace_size<double>(lda);
}
}  // namespace linalg
