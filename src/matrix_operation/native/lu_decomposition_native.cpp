#include "include/matrix_operation/lu_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "include/common/linalg_macros.h"

#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

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

#define A(i, j) m[(i) * lda + (j)]

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
template <typename T> inline void lu_axpy_sub(T* y, const T* x, T a, linalg_long len)
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

// Dot product of two length-`len` contiguous operands, used below for the
// trailing update's reduction. Vectorized the same way
// cholesky_decomposition.cpp's blocked kernel vectorizes its own (symmetric)
// trailing-update dot product, via linalg::packet<T>.
//
// Uses 4 independent accumulators (unlike lu_axpy_sub above, this *is* a
// true reduction, so a single accumulator is a genuine loop-carried FMA
// dependency chain -- the CPU can't start iteration N+1's FMA until
// iteration N's completes, stalling on FMA latency rather than throughput).
// Combined at the end via tree reduction before the horizontal predux.
template <typename T> inline T lu_dot(const T* x, const T* y, linalg_long len)
{
#if LINALG_HAS_VECTORIZATION
    using P                     = packet<T>;
    constexpr linalg_long width = P::size;

    typename P::packet_t acc0, acc1, acc2, acc3;
    P::set(static_cast<T>(0), acc0);
    P::set(static_cast<T>(0), acc1);
    P::set(static_cast<T>(0), acc2);
    P::set(static_cast<T>(0), acc3);

    linalg_long c = 0;
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
    for (linalg_long c = 0; c < len; ++c)
    {
        sum += x[c] * y[c];
    }
    return sum;
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
template <typename T>
bool lu_panel_factor(T*       m,
    linalg_long               lda,
    linalg_long               start,
    linalg_long               width,
    LINALG_UNUSED linalg_int* pivot,
    LINALG_UNUSED T           tolerance)
{
    using size_type = linalg_long;

    for (size_type ii = 0; ii < width; ++ii)
    {
        const size_type i = start + ii;
#if LINALG_HAS_LU_PIVOTING
        T         maxA = std::abs(A(i, i));
        size_type imax = i;
        for (size_type k = i + 1; k < lda; ++k)
        {
            const auto absA = std::fabs(A(k, i));
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
            for (size_type j = 0; j < start + width; ++j)
            {
                std::swap(A(i, j), A(imax, j));
            }
        }
#endif
        const auto norm = static_cast<T>(1. / A(i, i));
        T* const   l_i  = &A(i, start);

        for (size_type r = i + 1; r < lda; ++r)
        {
            T* const l_r = &A(r, start);
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
template <typename T>
void lu_apply_pivots_to_trailing(
    T* m, linalg_long lda, linalg_long start, linalg_long width, const linalg_int* pivot)
{
    using size_type = linalg_long;

    for (size_type i = start; i < start + width; ++i)
    {
        const size_type piv = static_cast<size_type>(pivot[i]) - 1;
        if (piv != i)
        {
            for (size_type j = start + width; j < lda; ++j)
            {
                std::swap(A(i, j), A(piv, j));
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
// L11, then fold L21 * U12 into the trailing submatrix with the packed,
// dot-product-based rank-kLuBlockSize update below before moving to the
// next panel. Degenerates to exactly the unblocked kernel above when
// lda <= kLuBlockSize.
inline constexpr linalg_long kLuBlockSize = 64;

// Column-tile width the trailing update below packs U12 into, a fixed-size
// stack buffer rather than caller-provided workspace (this library never
// allocates its own scratch -- see include/common/workspace_arena.h -- and
// kLuBlockSize * kLuPackTile is small and constant regardless of lda).
inline constexpr linalg_long kLuPackTile = 128;

template <typename T>
bool lu_decomposition_blocked_impl(T* m, linalg_long lda, linalg_int* pivot, T tolerance)
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

        if (!lu_panel_factor(m, lda, k, bs, pivot, tolerance))
        {
            return false;
        }

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
        for (size_type ii = 0; ii < bs; ++ii)
        {
            T* const row_i = &A(k + ii, k + bs);
            for (size_type p = 0; p < ii; ++p)
            {
                const T  l_ip  = A(k + ii, k + p);
                T* const row_p = &A(k + p, k + bs);
                lu_axpy_sub(row_i, row_p, l_ip, rest);
            }
        }

        // Trailing update: A22 -= L21 * U12. L21 (&A(k+bs, k), width bs) is
        // already row-contiguous, but U12's columns are not (U12 is stored
        // bs rows x rest columns, so a column is strided by lda) -- so a
        // per-output-element dot product of an L21 row against a U12
        // column cannot walk either operand contiguously without first
        // packing one of them. This packs a kLuPackTile-wide slice of U12,
        // transposed, into `packed` (bs rows become bs-contiguous columns),
        // then reduces each A22 element with one lu_dot call against that
        // packed column -- touching every A22 element exactly once,
        // instead of the `bs` separate read-modify-write passes a
        // broadcast-and-axpy rank-1 formulation would need. This is the
        // same role GEBP micro-kernel packing (multiplication.cpp's native
        // GEMM, Eigen's generic_pack_rhs) plays for a real Level-3 GEMM --
        // the structural reason LAPACK's dgetrf/Eigen's PartialPivLU defer
        // this update to a GEMM call in the first place.
        T packed[kLuBlockSize * kLuPackTile];
        for (size_type col0 = 0; col0 < rest; col0 += kLuPackTile)
        {
            const size_type cw = std::min(kLuPackTile, rest - col0);
            for (size_type c = 0; c < cw; ++c)
            {
                for (size_type p = 0; p < bs; ++p)
                {
                    packed[c * bs + p] = A(k + p, k + bs + col0 + c);
                }
            }

            for (size_type r = 0; r < rest; ++r)
            {
                const T* const l21_row = &A(k + bs + r, k);
                for (size_type c = 0; c < cw; ++c)
                {
                    const T* const u12_col = &packed[c * bs];
                    A(k + bs + r, k + bs + col0 + c) -= lu_dot(l21_row, u12_col, bs);
                }
            }
        }
    }
    return true;
}

#undef A

}  // namespace

template <>
bool lu_decomposition<float, linalg_backend::native>::apply(
    float* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/)
{
    check_lu_inputs(m, pivot, lda);
    return lu_decomposition_blocked_impl(m, lda, pivot, std::numeric_limits<float>::epsilon());
}

template <>
bool lu_decomposition<double, linalg_backend::native>::apply(
    double* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/)
{
    check_lu_inputs(m, pivot, lda);
    return lu_decomposition_blocked_impl(m, lda, pivot, std::numeric_limits<double>::epsilon());
}

template <>
size_t lu_decomposition<float, linalg_backend::native>::workspace_size(linalg_int /*lda*/)
{
    return 0;
}

template <>
size_t lu_decomposition<double, linalg_backend::native>::workspace_size(linalg_int /*lda*/)
{
    return 0;
}
}  // namespace linalg
