#include "include/matrix_operation/svd_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/workspace_arena.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#if LINALG_HAS_VECTORIZATION
#include <backend/simd.h>
#endif

#include "include/common/linalg_profiler.h"

namespace linalg
{

// Null/positivity checks shared by every svd_decomposition<T, Backend>::apply
// below, so callers (tests included) get the same validation regardless of
// which backend is active. There is no non-detail svd_decomposition()
// wrapper to put this in any more — callers use
// svd_decomposition<T, Backend>::apply directly, naming Backend explicitly.
static void check_svd_inputs(const void* A,
    const void*                          S,
    const void*                          U,
    const void*                          VT,
    linalg_long                          rows,
    linalg_long                          columns,
    linalg_long                          lda,
    linalg_long                          ldu,
    linalg_long                          ldv)
{
    LINALG_CHECK(A != nullptr && S != nullptr && U != nullptr && VT != nullptr,
        "svd_decomposition::apply: A, S, U, and VT must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldu != 0 && ldv != 0,
        "svd_decomposition::apply: rows/columns/lda/ldu/ldv must be positive");
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/svd_decomposition_mkl.cpp and
// blas/svd_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

// Dot product of two contiguous length-`len` operands, used below to
// compute a Householder dot-product against a row that is already
// contiguous in S_/U_ (the row Householder's S_ update and every column
// Householder's U_ update walk a row, not a strided column -- see SVD's
// comments at each call site). Same ::simd<T> (src/common/
// Scalars) SIMD abstraction, and the same 4-accumulator tree-reduction
// shape, as lu_decomposition_native.cpp's lu_dot.
// x and y are __restrict: every call site below passes a row of S_/U_
// against house_vec, two buffers svd_scalar_plan carves as disjoint
// sub-ranges of the caller's workspace arena, so they never alias.
template <typename T>
inline T svd_dot(const T* __restrict x, const T* __restrict y, linalg_long len)
{
#if LINALG_HAS_VECTORIZATION
    using P                     = simd<T>;
    constexpr linalg_long width = P::size;

    typename P::simd_t acc0, acc1, acc2, acc3;
    acc0 = P::set(static_cast<T>(0));
    acc1 = P::set(static_cast<T>(0));
    acc2 = P::set(static_cast<T>(0));
    acc3 = P::set(static_cast<T>(0));

    linalg_long c = 0;
    for (; c + 4 * width <= len; c += 4 * width)
    {
        typename P::simd_t px0, px1, px2, px3;
        typename P::simd_t py0, py1, py2, py3;
        px0  = P::loadu(x + c);
        px1  = P::loadu(x + c + width);
        px2  = P::loadu(x + c + 2 * width);
        px3  = P::loadu(x + c + 3 * width);
        py0  = P::loadu(y + c);
        py1  = P::loadu(y + c + width);
        py2  = P::loadu(y + c + 2 * width);
        py3  = P::loadu(y + c + 3 * width);
        acc0 = P::fma(px0, py0, acc0);
        acc1 = P::fma(px1, py1, acc1);
        acc2 = P::fma(px2, py2, acc2);
        acc3 = P::fma(px3, py3, acc3);
    }

    typename P::simd_t acc01, acc23, acc;
    acc01 = P::add(acc0, acc1);
    acc23 = P::add(acc2, acc3);
    acc   = P::add(acc01, acc23);

    for (; c + width <= len; c += width)
    {
        typename P::simd_t px;
        typename P::simd_t py;
        px  = P::loadu(x + c);
        py  = P::loadu(y + c);
        acc = P::fma(px, py, acc);
    }

    T sum = P::accumulate(acc);
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

// y[c] += alpha * x[c] for c in [0, len), both contiguous. Used both to
// apply a Householder reflection to an already-contiguous row (row_k[j] -=
// dot_prod * house_vec[j], i.e. alpha = -dot_prod, x = house_vec) and, via
// loop interchange, to turn a strided column update into a sum of
// contiguous row contributions (see the column-Householder S_/V_ updates
// in SVD below: accumulating w[k] = sum_j house_vec[j] * row_j[k] one row
// at a time is exactly `svd_axpy(w, row_j, house_vec[j], trailing)`).
// Mirrors qr_decomposition_native.cpp's qr_axpy.
// y and x are __restrict for the same reason as svd_dot above: call sites
// pass disjoint rows of S_/U_/V_, the `w` scratch row, or house_vec, never
// the same buffer for both.
template <typename T>
inline void svd_axpy(T* __restrict y, const T* __restrict x, T alpha, linalg_long len)
{
#if LINALG_HAS_VECTORIZATION
    using P                     = simd<T>;
    constexpr linalg_long width = P::size;

    typename P::simd_t pa;
    pa = P::set(alpha);

    linalg_long c = 0;
    for (; c + width <= len; c += width)
    {
        typename P::simd_t px;
        typename P::simd_t py;
        px = P::loadu(x + c);
        py = P::loadu(y + c);
        py = P::fma(pa, px, py);
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

// GivensL/GivensR below are deliberately generic over which of U_/S_/V_ is
// passed in (see their call sites in SVD below) with `dim` always the
// *caller's* local dim array for that buffer, not a fixed global -- the old
// U(i,j)/S(i,j)/V(i,j) macros captured that genericity by name (expanding
// against whatever local variable happened to be called `S_`/`dim` in
// scope), which is exactly the kind of implicit, name-dependent aliasing
// that breaks silently if a parameter is ever renamed. Both are rewritten
// below to index the parameters directly.

// GivensL zeroes a subdiagonal pair by rotating rows m and m+1 together;
// row_m/row_m1 are hoisted once (both rows are fixed for the whole loop,
// only the column i varies) instead of recomputing m * dim[1] and
// (m + 1) * dim[1] on every iteration. Both rows are contiguous (dim[1]
// elements each), unlike GivensR's column pair below, so this is
// vectorized via ::simd<T>: each lane independently computes the
// same 2x2 rotation per lane -- out0 = c*S0 - s*S1, out1 = s*S0 + c*S1 --
// expressed as two fma's per output since simd<T> has no standalone
// multiply (only fma/add), matching the rest of this file's idiom.
template <class T> void GivensL(T* S_, const linalg_long dim[2], linalg_long m, T a, T b)  // NOLINT
{
    T r = sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

    // row_m and row_m1 are always different rows (m + 1 != m), so safe to
    // mark __restrict relative to each other.
    T* const __restrict row_m  = S_ + m * dim[1];
    T* const __restrict row_m1 = row_m + dim[1];

#if LINALG_HAS_VECTORIZATION
    using P                     = simd<T>;
    constexpr linalg_long width = P::size;

    typename P::simd_t vc, vs, vnegs, vzero;
    vc    = P::set(c);
    vs    = P::set(s);
    vnegs = P::set(-s);
    vzero = P::set(static_cast<T>(0));

    linalg_long i = 0;
    for (; i + width <= dim[1]; i += width)
    {
        typename P::simd_t vS0, vS1;
        vS0 = P::loadu(row_m + i);
        vS1 = P::loadu(row_m1 + i);

        typename P::simd_t out0, out1, tmp0, tmp1;
        tmp0 = P::fma(vc, vS0, vzero);
        out0 = P::fma(vnegs, vS1, tmp0);
        tmp1 = P::fma(vs, vS0, vzero);
        out1 = P::fma(vc, vS1, tmp1);

        P::storeu(out0, row_m + i);
        P::storeu(out1, row_m1 + i);
    }
    for (; i < dim[1]; i++)
    {
        const T S0 = row_m[i];
        const T S1 = row_m1[i];
        row_m[i]   = c * S0 - s * S1;
        row_m1[i]  = s * S0 + c * S1;
    }
#else
    for (linalg_int i = 0; i < static_cast<linalg_int>(dim[1]); i++)
    {
        const T S0 = row_m[i];
        const T S1 = row_m1[i];
        row_m[i]   = c * S0 - s * S1;
        row_m1[i]  = s * S0 + c * S1;
    }
#endif
}

// GivensR is GivensL's column counterpart (rotates columns m and m+1 of
// row i together); row_i is hoisted once per i -- it's read/written at both
// column m and m + 1 -- instead of recomputing i * dim[1] twice per
// iteration. Left scalar, unlike GivensL above: each row only ever touches
// two elements here (columns m and m + 1), so there is no same-row run of
// contiguous elements for simd<T> to load/store over -- vectorizing
// would mean gathering across the `dim[0]` rows instead, a different and
// more involved transformation not worth it for this O(dim[0]) step.
template <class T> void GivensR(T* S_, const linalg_long dim[2], linalg_long m, T a, T b)  // NOLINT
{
    T r = sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

    for (linalg_int i = 0; i < static_cast<linalg_int>(dim[0]); i++)
    {
        T* const row_i = S_ + i * dim[1];
        const T  S0    = row_i[m];
        const T  S1    = row_i[m + 1];
        row_i[m]       = c * S0 - s * S1;
        row_i[m + 1]   = s * S0 + c * S1;
    }
}

// `house_vec` (max(dim[0],dim[1]) elements) and `w` (dim[1] elements, the
// column-Householder row-accumulation scratch -- see its use below) are
// caller-owned scratch. U_, S_, V_, house_vec, and w are __restrict:
// svd_scalar_plan carves all five as separate, disjoint sub-ranges of the
// caller's workspace arena (see svd() below), so they never overlap each
// other. Locals derived from them inside this function are deliberately
// left unmarked: several (e.g. row_n2 and row_k0 in the diagonalization
// step below) can legitimately address the same row depending on loop
// state, so blanket-marking every such view would be incorrect rather than
// just unhelpful.
template <class T>
void SVD(const linalg_long dim[2],
    T* __restrict U_,
    T* __restrict S_,
    T* __restrict V_,
    T* __restrict house_vec,
    T* __restrict w,
    T eps = -1)  // NOLINT
{
    assert(dim[0] >= dim[1]);

    {  // Bi-diagonalization
        const auto n = std::min(dim[0], dim[1]);
        for (linalg_long i = 0; i < n; i++)
        {
            // Column Householder. S(i, i) is read twice below (the pivot
            // magnitude and its sign); s_ii hoists that into one read. The
            // j-loops walk column i of S_ (stride dim[1]), so col_ptr steps
            // by dim[1] per iteration instead of each loop recomputing
            // j * dim[1] + i from scratch.
            {
                const T s_ii = S_[i * dim[1] + i];
                T       x1   = s_ii < 0 ? -s_ii : s_ii;

                T        x_inv_norm = 0;
                const T* col_ptr    = S_ + i * dim[1] + i;
                for (linalg_long j = i; j < dim[0]; j++, col_ptr += dim[1])
                {
                    x_inv_norm += (*col_ptr) * (*col_ptr);
                }
                if (x_inv_norm > 0)
                {
                    // Non-degenerate subcolumn: build the unit Householder
                    // vector as usual.
                    x_inv_norm = 1 / sqrt(x_inv_norm);

                    T alpha = sqrt(1 + x1 * x_inv_norm);
                    T beta  = x_inv_norm / alpha;

                    house_vec[i] = -alpha;
                    col_ptr      = S_ + (i + 1) * dim[1] + i;
                    for (linalg_long j = i + 1; j < dim[0]; j++, col_ptr += dim[1])
                    {
                        house_vec[j] = -beta * (*col_ptr);
                    }
                    if (s_ii < 0)
                    {
                        for (linalg_long j = i + 1; j < dim[0]; j++)
                        {
                            house_vec[j] = -house_vec[j];
                        }
                    }
                }
                else
                {
                    // Subcolumn (including the pivot) is already exactly
                    // zero: there is nothing to reflect. Falling through to
                    // alpha = sqrt(1 + 0) = 1 here would fabricate a
                    // spurious non-identity reflection (house_vec[i] = -1)
                    // that corrupts an already-zero entry instead of
                    // leaving it alone — this hit e.g. the identity matrix,
                    // where the bidiagonalization has nothing left to do.
                    for (linalg_long j = i; j < dim[0]; j++)
                    {
                        house_vec[j] = 0;
                    }
                }
            }
            // Apply the column reflection to S_'s trailing columns
            // [i, dim[1)). Per output column k this is a dot product and
            // an update that both stride across rows of S_ (column k,
            // stride dim[1]) -- exactly what qr_decomposition_native.cpp's
            // forward pass avoids via a GEMV-then-GER loop interchange
            // (see its function comment), applied here the same way:
            // w[k - i] = sum_j house_vec[j] * S_[j][k] is accumulated one
            // row of S_ at a time (contiguous, via svd_axpy), then S_[j][k]
            // -= house_vec[j] * w[k - i] is applied the same way. house_vec
            // rows that are exactly zero (common once the subcolumn below
            // row i + 1 has already been eliminated) are skipped rather
            // than axpy'd with a zero scalar.
            {
                const linalg_long trailing = dim[1] - i;
                std::fill_n(w, static_cast<size_t>(trailing), static_cast<T>(0));
                for (linalg_long j = i; j < dim[0]; j++)
                {
                    if (house_vec[j] != 0)
                    {
                        svd_axpy(w, S_ + j * dim[1] + i, house_vec[j], trailing);
                    }
                }
                for (linalg_long j = i; j < dim[0]; j++)
                {
                    if (house_vec[j] != 0)
                    {
                        svd_axpy(S_ + j * dim[1] + i, w, -house_vec[j], trailing);
                    }
                }
            }
            // Apply the same column reflection to U_. Unlike S_ above,
            // U_'s rows (row_k, contiguous since U_ is indexed
            // [row * dim[0] + col]) already line up with house_vec, so no
            // loop interchange is needed: each row's dot product and
            // update are already contiguous, vectorized directly via
            // svd_dot/svd_axpy.
            for (linalg_int k = 0; k < static_cast<linalg_int>(dim[0]); k++)
            {
                T* const row_k    = U_ + k * dim[0];
                const T  dot_prod = svd_dot(row_k + i, house_vec + i, dim[0] - i);
                svd_axpy(row_k + i, house_vec + i, -dot_prod, dim[0] - i);
            }

            // Row Householder
            if (i >= n - 1)
            {
                continue;
            }
            // Row Householder. row_i (S_'s row i, contiguous) is hoisted
            // once and reused for every column access below, including the
            // doubled read of S(i, i + 1) (s_i_i1).
            {
                T* const row_i  = S_ + i * dim[1];
                const T  s_i_i1 = row_i[i + 1];
                T        x1     = s_i_i1 < 0 ? -s_i_i1 : s_i_i1;

                // row_i[i+1 .. dim[1]) is contiguous, unlike the column
                // Householder's subcolumn norm above, so its self dot
                // product is vectorized via svd_dot instead of a plain
                // scalar accumulation.
                T x_inv_norm = svd_dot(row_i + i + 1, row_i + i + 1, dim[1] - i - 1);
                if (x_inv_norm > 0)
                {
                    // Non-degenerate subrow: build the unit Householder
                    // vector as usual.
                    x_inv_norm = 1 / sqrt(x_inv_norm);

                    T alpha = sqrt(1 + x1 * x_inv_norm);
                    T beta  = x_inv_norm / alpha;

                    house_vec[i + 1] = -alpha;
                    for (linalg_long j = i + 2; j < dim[1]; j++)
                    {
                        house_vec[j] = -beta * row_i[j];
                    }
                    if (s_i_i1 < 0)
                    {
                        for (linalg_long j = i + 2; j < dim[1]; j++)
                        {
                            house_vec[j] = -house_vec[j];
                        }
                    }
                }
                else
                {
                    // Subrow (including the pivot) is already exactly
                    // zero — see the matching column-Householder branch
                    // above for why this must be a no-op rather than
                    // falling through to a spurious alpha = 1 reflection.
                    for (linalg_long j = i + 1; j < dim[1]; j++)
                    {
                        house_vec[j] = 0;
                    }
                }
            }
            // Apply the row reflection to S_'s rows [i, dim[0)). Each
            // row_k (S_'s row k) is already contiguous over the columns
            // house_vec addresses, so -- unlike V_'s column update below
            // -- no loop interchange is needed: dot product and update are
            // both vectorized directly via svd_dot/svd_axpy.
            for (auto k = static_cast<linalg_int>(i); k < static_cast<linalg_int>(dim[0]); k++)
            {
                T* const row_k    = S_ + k * dim[1];
                const T  dot_prod = svd_dot(row_k + i + 1, house_vec + i + 1, dim[1] - i - 1);
                svd_axpy(row_k + i + 1, house_vec + i + 1, -dot_prod, dim[1] - i - 1);
            }
            // Apply the row reflection to V_'s columns [0, dim[1)), each
            // of which strides across rows of V_ -- the same shape as the
            // column Householder's S_ update above, so it uses the same
            // GEMV-then-GER loop interchange: accumulate
            // w[k] = sum_j house_vec[j] * V_[j][k] one row of V_ at a time
            // (contiguous, via svd_axpy over the full dim[1] columns),
            // then apply V_[j][k] -= house_vec[j] * w[k] the same way.
            {
                std::fill_n(w, static_cast<size_t>(dim[1]), static_cast<T>(0));
                for (linalg_long j = i + 1; j < dim[1]; j++)
                {
                    if (house_vec[j] != 0)
                    {
                        svd_axpy(w, V_ + j * dim[1], house_vec[j], dim[1]);
                    }
                }
                for (linalg_long j = i + 1; j < dim[1]; j++)
                {
                    if (house_vec[j] != 0)
                    {
                        svd_axpy(V_ + j * dim[1], w, -house_vec[j], dim[1]);
                    }
                }
            }
        }
    }

    linalg_long k0 = 0;
    if (eps < 0)
    {
        eps = 1.0;
        while (eps + (T)1.0 > 1.0)
        {
            eps *= 0.5;
        }
        eps *= 64.0;
    }
    while (k0 < dim[1] - 1)
    {  // Diagonalization
        // S_max walks the diagonal, stride dim[1] + 1, via a running
        // pointer instead of recomputing i * dim[1] + i on every i.
        T        S_max    = 0.0;
        const T* diag_ptr = S_;
        for (linalg_long i = 0; i < dim[1]; i++, diag_ptr += dim[1] + 1)
        {
            S_max = (S_max > *diag_ptr ? S_max : *diag_ptr);
        }

        while (k0 < dim[1] - 1 && fabs(S_[k0 * dim[1] + k0 + 1]) <= eps * S_max)
        {
            k0++;
        }
        if (k0 == dim[1] - 1)
        {
            continue;
        }

        linalg_long n = k0 + 2;
        while (n < dim[1] && fabs(S_[(n - 1) * dim[1] + n]) > eps * S_max)
        {
            n++;
        }

        T alpha = 0;
        T beta  = 0;
        {
            // Compute mu. row_n2 (row n - 2) is read at both column n - 2
            // and n - 1 below, so it's hoisted once rather than recomputed
            // for each.
            T C[2][2];  // NOLINT

            const T* const row_n2 = S_ + (n - 2) * dim[1];
            const T        s_n2_2 = row_n2[n - 2];
            const T        s_n2_1 = row_n2[n - 1];
            const T        s_n1_1 = S_[(n - 1) * dim[1] + (n - 1)];

            C[0][0] = s_n2_2 * s_n2_2;
            if (n - k0 > 2)
            {
                const T s_n3_2 = S_[(n - 3) * dim[1] + (n - 2)];
                C[0][0] += s_n3_2 * s_n3_2;
            }
            C[0][1] = s_n2_2 * s_n2_1;
            C[1][0] = s_n2_2 * s_n2_1;
            C[1][1] = s_n1_1 * s_n1_1 + s_n2_1 * s_n2_1;

            T b = -(C[0][0] + C[1][1]) / 2;
            T c = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            T d = sqrt(b * b - c);

            T lambda1 = -b + d;
            T lambda2 = -b - d;

            T d1 = std::fabs(lambda1 - C[1][1]);
            T d2 = std::fabs(lambda2 - C[1][1]);
            T mu = (d1 < d2 ? lambda1 : lambda2);

            // row_k0 (row k0) is read at columns k0 and k0 + 1; s_k0_k0
            // itself is used twice (squared), so it's hoisted into one
            // read too.
            const T* const row_k0  = S_ + k0 * dim[1];
            const T        s_k0_k0 = row_k0[k0];
            alpha                  = s_k0_k0 * s_k0_k0 - mu;
            beta                   = s_k0_k0 * row_k0[k0 + 1];
        }

        for (linalg_long k = k0; k < n - 1; k++)
        {
            linalg_long dimU[2] = {dim[0], dim[0]};  // NOLINT
            linalg_long dimV[2] = {dim[1], dim[1]};  // NOLINT
            GivensR(S_, dim, k, alpha, beta);
            GivensL(V_, dimV, k, alpha, beta);

            // row_k (row k of S_, just rotated by the GivensR/GivensL pair
            // above) is read at columns k, k + 1, and k + 2 below; hoisted
            // once and reused for all three instead of recomputing
            // k * dim[1] each time. It stays valid across the GivensL/
            // GivensR calls between its first and later uses since it's
            // only an address into S_'s storage, not a cached value.
            const T* const row_k = S_ + k * dim[1];
            alpha                = row_k[k];
            beta                 = S_[(k + 1) * dim[1] + k];
            GivensL(S_, dim, k, alpha, beta);
            GivensR(U_, dimU, k, alpha, beta);

            alpha = row_k[k + 1];
            beta  = row_k[k + 2];
        }

        {  // rowsake S bi-diagonal again
            for (linalg_long i0 = k0; i0 < n - 1; i0++)
            {
                T* const row_i0 = S_ + i0 * dim[1];
                for (linalg_long i1 = 0; i1 < dim[1]; i1++)
                {
                    if (i0 > i1 || i0 + 1 < i1)
                    {
                        row_i0[i1] = 0;
                    }
                }
            }
            for (linalg_long i0 = 0; i0 < dim[0]; i0++)
            {
                T* const row_i0 = S_ + i0 * dim[1];
                for (linalg_long i1 = k0; i1 < n - 1; i1++)
                {
                    if (i0 > i1 || i0 + 1 < i1)
                    {
                        row_i0[i1] = 0;
                    }
                }
            }
            for (linalg_long i = 0; i < dim[1] - 1; i++)
            {
                T* const entry = &S_[i * dim[1] + (i + 1)];
                if (fabs(*entry) <= eps * S_max)
                {
                    *entry = 0;
                }
            }
        }
    }
}

template <typename T> struct svd_scalar_layout
{
    T* U_;
    T* V_;
    T* S_;
    T* house_vec;
    T* w;
};

template <typename T>
svd_scalar_layout<T> svd_scalar_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns)
{
    const linalg_long    dim[2] = {std::max(rows, columns), std::min(rows, columns)};  // NOLINT
    svd_scalar_layout<T> p;
    p.U_        = arena.next<T>(static_cast<size_t>(dim[0]) * static_cast<size_t>(dim[0]));
    p.V_        = arena.next<T>(static_cast<size_t>(dim[1]) * static_cast<size_t>(dim[1]));
    p.S_        = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.house_vec = arena.next<T>(static_cast<size_t>(std::max(dim[0], dim[1])));
    // Row-accumulation scratch for the column-Householder S_/V_ updates in
    // SVD below (the GEMV-then-GER loop interchange that replaces a
    // strided column dot-product/update with contiguous row axpy's --
    // see those call sites). Sized to the wider of the two uses: up to
    // dim[1] columns for S_'s trailing-column update, the full dim[1]
    // columns for V_'s.
    p.w = arena.next<T>(static_cast<size_t>(dim[1]));
    return p;
}

template <typename T> size_t svd_scalar_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    svd_scalar_plan<T>(arena, rows, columns);
    return arena.total_size();
}

// A, S, U, and VT are __restrict: svd_decomposition.h documents them as
// four distinct caller-owned buffers (A is read-only input; S, U, VT are
// separate outputs), and this function never writes through one while
// reading another's memory -- the arena-carved U_/S_/V_/house_vec scratch
// is itself disjoint from all four (see svd_scalar_plan/SVD above).
template <class T>
inline void svd(  // NOLINT
    linalg_long rows,
    linalg_long columns,
    T* __restrict A,
    linalg_long lda,
    T* __restrict S,
    T* __restrict U,
    linalg_long ldu,
    T* __restrict VT,
    linalg_long ldv,
    void*       workspace,
    size_t      workspace_size)
{
    const linalg_long dim[2] = {std::max(rows, columns), std::min(rows, columns)};  // NOLINT

    LINALG_CHECK(workspace_size >= svd_scalar_workspace_size_impl<T>(rows, columns),
        "svd_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = svd_scalar_plan<T>(arena, rows, columns);
    auto* const             U_   = bufs.U_;
    auto* const             V_   = bufs.V_;
    auto* const             S_   = bufs.S_;

    memset(U_, 0, dim[0] * dim[0] * sizeof(T));
    memset(V_, 0, dim[1] * dim[1] * sizeof(T));

    if (dim[1] == columns)
    {
        for (linalg_long i = 0; i < dim[0]; i++)
        {
            for (linalg_long j = 0; j < dim[1]; j++)
            {
                S_[i * dim[1] + j] = A[i * lda + j];
            }
        }
    }
    else
    {
        for (linalg_long i = 0; i < dim[0]; i++)
        {
            for (linalg_long j = 0; j < dim[1]; j++)
            {
                S_[i * dim[1] + j] = A[j * lda + i];
            }
        }
    }

    for (linalg_long i = 0; i < dim[0]; i++)
    {
        U_[i * dim[0] + i] = 1;
    }
    for (linalg_long i = 0; i < dim[1]; i++)
    {
        V_[i * dim[1] + i] = 1;
    }

    SVD<T>(dim, U_, S_, V_, bufs.house_vec, bufs.w, (T)-1);

    for (linalg_long i = 0; i < dim[1]; i++)
    {  // Set S
        S[i] = S_[i * dim[1] + i];
    }
    if (dim[1] == columns)
    {  // A was factored directly: U_ holds left vectors, V_ holds VT.
        for (linalg_long i = 0; i < rows; i++)
        {
            for (linalg_long j = 0; j < columns; j++)
            {
                U[j + ldu * i] = U_[j + i * rows];
            }
        }
        for (linalg_long i = 0; i < columns; i++)
        {
            for (linalg_long j = 0; j < columns; j++)
            {
                VT[j + ldv * i] = V_[j + i * columns] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
            }
        }
    }
    else
    {  // A was transposed before the factorization, so the factors swap roles.
        for (linalg_long i = 0; i < rows; i++)
        {
            for (linalg_long j = 0; j < rows; j++)
            {
                U[j + ldu * i] = V_[i + j * rows];
            }
        }
        for (linalg_long i = 0; i < rows; i++)
        {
            for (linalg_long j = 0; j < columns; j++)
            {
                VT[j + ldv * i] = U_[i + j * columns] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
            }
        }
    }

    for (linalg_long i = 0; i < dim[1]; i++)
    {
        S[i] = S[i] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
    }
}

}  // namespace

template <>
void svd_decomposition<float, linalg_backend::NATIVE>::apply(linalg_long rows,
    linalg_long                                                          columns,
    float* __restrict A,
    linalg_long lda,
    float* __restrict S,
    float* __restrict U,
    linalg_long ldu,
    float* __restrict VT,
    linalg_long ldv,
    void*       workspace,
    size_t      workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("svd_decomposition/native<float>");
    check_svd_inputs(A, S, U, VT, rows, columns, lda, ldu, ldv);
    svd(rows, columns, A, lda, S, U, ldu, VT, ldv, workspace, workspace_size);
}

template <>
void svd_decomposition<double, linalg_backend::NATIVE>::apply(linalg_long rows,
    linalg_long                                                           columns,
    double* __restrict A,
    linalg_long lda,
    double* __restrict S,
    double* __restrict U,
    linalg_long ldu,
    double* __restrict VT,
    linalg_long ldv,
    void*       workspace,
    size_t      workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("svd_decomposition/native<double>");
    check_svd_inputs(A, S, U, VT, rows, columns, lda, ldu, ldv);
    svd(rows, columns, A, lda, S, U, ldu, VT, ldv, workspace, workspace_size);
}

template <>
size_t svd_decomposition<float, linalg_backend::NATIVE>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return svd_scalar_workspace_size_impl<float>(rows, columns);
}

template <>
size_t svd_decomposition<double, linalg_backend::NATIVE>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return svd_scalar_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg
