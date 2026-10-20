#include "include/matrix_operation/cholesky_decomposition.h"

#include "include/common/backend.h"
#include "include/common/linalg_checks.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

#include "include/common/linalg_profiler.h"

namespace linalg
{
namespace detail
{

// Null/positivity checks shared by every cholesky_decomposition<T, Backend>::apply
// below, so callers (linear_solver.cpp, inversion.cpp, this file's
// own scalar path) get the same validation regardless of which backend is
// active. There is no non-detail cholesky_decomposition() wrapper to put
// this in any more — callers use cholesky_decomposition<T, Backend>::apply
// directly, naming Backend explicitly.
static void check_cholesky_inputs(const void* C, linalg_int lda)
{
    LINALG_CHECK(C != nullptr, "cholesky_decomposition::apply: C must not be null");
    LINALG_CHECK(lda > 0, "cholesky_decomposition::apply: lda must be positive", lda);
}

}  // namespace detail

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/cholesky_decomposition_mkl.cpp and
// blas/cholesky_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

// Dot product of two length-`len` contiguous rows, used below by the unblocked
// diagonal-block kernel's own forward-substitution and norm reductions, and by
// the blocked kernel for the panel solve's forward-substitution reduction and
// the trailing submatrix's rank update -- blocking turns all of these into
// fixed-shape reductions worth vectorizing explicitly. Uses linalg::packet<T>
// (src/common/packet.h), the same Eigen-derived SIMD abstraction
// multiplication.cpp's native GEMM kernel is built on, instead of hoping the
// compiler auto-vectorizes a scalar loop. Four independent accumulators,
// combined via tree reduction before the horizontal predux, hide FMA latency
// on the loop-carried dependency a single accumulator would create -- the
// same fix lu_decomposition_native.cpp's lu_dot applies, for the same reason.
template <typename T> inline T cholesky_dot(const T* x, const T* y, linalg_int len)
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
    T sum = 0;
    for (linalg_int c = 0; c < len; ++c)
    {
        sum += x[c] * y[c];
    }
    return sum;
#endif
}

// Unblocked, row-major Cholesky of the lower triangle (A = L * L^T): the base
// case the blocked kernel below falls back to for one kCholeskyBlockSize-wide
// diagonal block (or the whole matrix, when n <= kCholeskyBlockSize). Only
// ever instantiated for LOWER_TRIANGULAR, row-major -- the apply()
// specializations below get UPPER_TRIANGULAR by factoring into the lower
// triangle and transposing, and this file has no col-major caller -- so
// neither is a template parameter here.
template <typename T>
linalg_int cholesky_decomposition_scalar_impl(linalg_int n, T* A, linalg_int lda)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        auto* a_i = &A[i * lda];

        for (linalg_int j = 0; j < i; ++j)
        {
            auto*   a_j = &A[j * lda];
            const T sum = a_i[j] - cholesky_dot(a_i, a_j, j);
            a_i[j]      = sum / a_j[j];
            a_j[i]      = static_cast<T>(0);
        }

        const T sum = a_i[i] - cholesky_dot(a_i, a_i, i);

        if (sum <= 0)
        {
            return i + 1;  // Matrix is not positive-definite
        }

        a_i[i] = std::sqrt(sum);
    }

    return 0;  // Success
}

// Right-looking blocked Cholesky, the same shape PyTorch's CPU path gets for free by
// calling LAPACK's (recursive, blocked) potrf, and that Eigen's LLT (Eigen/src/Cholesky/
// LLT.h) implements directly: factor a kCholeskyBlockSize-wide diagonal block with the
// unblocked kernel above, solve the panel below it against that block (forward
// substitution against L11), then fold the panel's outer product into the trailing
// submatrix before moving to the next block. Blocking keeps each diagonal block --
// and every panel row solved against it -- resident in cache instead of walking the
// full matrix row by row, and turns the trailing-submatrix reduction into
// kCholeskyBlockSize-length, fixed-trip-count inner products the compiler can
// auto-vectorize, instead of one growing-length reduction per (i, j) pair. Degenerates
// to exactly the unblocked kernel above when n <= kCholeskyBlockSize. Produces L with
// A = L * L^T; the apply() specializations below reuse it for UPPER_TRIANGULAR too,
// by factoring into the lower triangle and transposing, for the same row-contiguous-
// access reason the old UPPER branch here did.
inline constexpr linalg_int kCholeskyBlockSize = 64;

template <typename T>
linalg_int cholesky_decomposition_blocked_lower(linalg_int n, T* A, linalg_int lda)
{
    for (linalg_int k = 0; k < n; k += kCholeskyBlockSize)
    {
        const linalg_int bs  = std::min(kCholeskyBlockSize, n - k);
        T* const         l11 = &A[k * lda + k];

        const linalg_int rc = cholesky_decomposition_scalar_impl<T>(bs, l11, lda);
        if (rc != 0)
        {
            return k + rc;
        }

        // The unblocked diagonal-block kernel only zeroes the strict upper triangle
        // within that bs x bs block; zero the rest of these rows' upper triangle here
        // (columns in later blocks), so the full output matches the documented
        // all-zeros-above-the-diagonal contract regardless of block boundaries.
        for (linalg_int i = k; i < k + bs; ++i)
        {
            std::fill_n(&A[i * lda + k + bs], n - k - bs, static_cast<T>(0));
        }

        const linalg_int rows_below = n - k - bs;
        if (rows_below == 0)
        {
            continue;
        }

        T* const a21 = &A[(k + bs) * lda + k];
        T* const a22 = &A[(k + bs) * lda + (k + bs)];

        // Panel solve: L21 = A21 * L11^-T, one row of A21 at a time via forward
        // substitution against the already-factored L11 (L11 * L21[r,:]^T = A21[r,:]^T).
        for (linalg_int r = 0; r < rows_below; ++r)
        {
            T* const row = &a21[r * lda];
            for (linalg_int j = 0; j < bs; ++j)
            {
                const T* const l_j = &l11[j * lda];
                const T        sum = row[j] - cholesky_dot(row, l_j, j);
                row[j]             = sum / l_j[j];
            }
        }

        // Trailing update: A22 -= L21 * L21^T, lower triangle only -- the upper
        // triangle of A22 is never read by this algorithm.
        for (linalg_int i = 0; i < rows_below; ++i)
        {
            const T* const row_i = &a21[i * lda];
            T* const       a22_i = &a22[i * lda];
            for (linalg_int j = 0; j <= i; ++j)
            {
                const T* const row_j = &a21[j * lda];
                a22_i[j] -= cholesky_dot(row_i, row_j, bs);
            }
        }
    }

    return 0;  // Success
}

template <typename T>
bool cholesky_decomposition_apply_impl(T* C, linalg_int n, cholesky_decomposition_enum type)
{
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return cholesky_decomposition_blocked_lower<T>(n, C, n) == 0;
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
    {
        if (cholesky_decomposition_blocked_lower<T>(n, C, n) != 0)
        {
            return false;
        }
        for (linalg_int i = 0; i < n; ++i)
        {
            for (linalg_int j = 0; j < i; ++j)
            {
                C[j * n + i] = C[i * n + j];
                C[i * n + j] = 0.;
            }
        }
        return true;
    }
    default:
        LINALG_THROW("Unsupported enum type!");
    }
}

}  // namespace

template <>
bool cholesky_decomposition<float, linalg_backend::NATIVE>::apply(
    float* C, linalg_int lda, cholesky_decomposition_enum type, void* /*stream*/)
{
    LINALG_SCOPE("cholesky_decomposition/native<float>");
    detail::check_cholesky_inputs(C, lda);
    return cholesky_decomposition_apply_impl<float>(C, lda, type);
}

template <>
bool cholesky_decomposition<double, linalg_backend::NATIVE>::apply(
    double* C, linalg_int lda, cholesky_decomposition_enum type, void* /*stream*/)
{
    LINALG_SCOPE("cholesky_decomposition/native<double>");
    detail::check_cholesky_inputs(C, lda);
    return cholesky_decomposition_apply_impl<double>(C, lda, type);
}

// Reverse-mode adjoint of cholesky_decomposition<T, Backend>::apply_aad: no vendor
// library (LAPACK/cuSOLVER) exposes this, so it always runs the scalar
// implementation directly, regardless of which backend is active.
namespace detail
{
// Row-major only -- this file has no col-major caller -- so unlike `part`
// (both LOWER_TRIANGULAR and UPPER_TRIANGULAR are real, used instantiations
// below), layout is not a template parameter here.
template <typename T, cholesky_decomposition_enum part>
void cholesky_decomposition_aad_impl(linalg_int n, T* C_aad, const T* C, linalg_long lda, T* A_aad)
{
    if constexpr (part == cholesky_decomposition_enum::LOWER_TRIANGULAR)  // NOLINT
    {
        for (linalg_int i = n - 1; i >= 0; --i)
        {
            const auto* const l_i = &C[i * lda];

            auto* l_i_aad = &C_aad[i * lda];
            auto* a_i_aad = &A_aad[i * lda];
            {
                // AAD: l_i[i] = std::sqrt(sum);
                auto sum_aad = 0.5 * l_i_aad[i] / l_i[i];
                // l_i_aad[i]   = 0.;

                for (linalg_int k = 0; k < i; ++k)
                {
                    // AAD: sum -= l_i[k] * l_i[k];
                    l_i_aad[k] -= static_cast<T>(2. * sum_aad * l_i[k]);
                }

                // AAD: auto sum = a_i[i];
                a_i_aad[i] += static_cast<T>(sum_aad);
            }
            for (linalg_int j = i - 1; j >= 0; --j)
            {
                const auto* const l_j     = &C[j * lda];
                auto*             l_j_aad = &C_aad[j * lda];

                auto diag = l_j[j];
                if (diag != 0.)
                {
                    // AAD: l_i[j] = sum / l_j[j];
                    double sum_aad = l_i_aad[j] / diag;
                    l_j_aad[j] -= static_cast<T>(sum_aad * l_i[j]);
                    l_i_aad[j] = 0.;

                    for (linalg_int k = 0; k < j; ++k)
                    {
                        // AAD: sum -= l_i[k] * l_j[k];
                        l_i_aad[k] -= static_cast<T>(sum_aad * l_j[k]);
                        l_j_aad[k] -= static_cast<T>(sum_aad * l_i[k]);
                    }

                    // AAD: auto sum = a_i[j];
                    a_i_aad[j] += static_cast<T>(sum_aad);
                }
            }
        }
    }
    else if constexpr (part == cholesky_decomposition_enum::UPPER_TRIANGULAR)
    {
        for (linalg_int i = n - 1; i >= 0; --i)
        {
            const auto* const u_i = &C[i * lda];

            const auto* u_i_aad = &C_aad[i * lda];
            auto*       a_i_aad = &A_aad[i * lda];
            {
                // AAD: u_i[i] = std::sqrt(sum);
                auto sum_aad = 0.5 * u_i_aad[i] / u_i[i];
                // u_i_aad[i]   = 0.;

                for (linalg_int k = 0; k < i; ++k)
                {
                    // AAD: sum -=  C[k * lda + i] *  C[k * lda + i];
                    C_aad[k * lda + i] -= static_cast<T>(2. * sum_aad * C[k * lda + i]);
                }

                // AAD: auto sum = a_i[i];
                a_i_aad[i] += static_cast<T>(sum_aad);
            }
            for (linalg_int j = i - 1; j >= 0; --j)
            {
                const auto* const u_j     = &C[j * lda];
                auto*             u_j_aad = &C_aad[j * lda];
                auto*             a_j_aad = &A_aad[j * lda];

                auto diag = u_j[j];
                if (diag != 0.)
                {
                    // AAD: u_j[i] = sum / u_j[j];
                    auto sum_aad = u_j_aad[i] / diag;
                    u_j_aad[j] -= sum_aad * u_j[i];
                    u_j_aad[i] = 0.;

                    for (linalg_int k = 0; k < j; ++k)
                    {
                        // AAD: sum -=  C[k * lda + i] *  C[k * lda + j];
                        C_aad[k * lda + i] -= sum_aad * C[k * lda + j];
                        C_aad[k * lda + j] -= sum_aad * C[k * lda + i];
                    }

                    // AAD: auto sum = a_j[i];
                    a_j_aad[i] += sum_aad;
                }
            }
        }
    }
    else
    {
        throw std::runtime_error("Invalid MatrixPart specified");
    }
}

}  // namespace detail

//-----------------------------------------------------------------------------
// apply_aad has no vendor-library equivalent: the mkl and blas_lapacke
// specializations forward to these native ones.
template <>
bool cholesky_decomposition<float, linalg_backend::NATIVE>::apply_aad(
    float* C_aad, const float* C, linalg_int lda, cholesky_decomposition_enum type, float* A_aad)
{
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        detail::cholesky_decomposition_aad_impl<float,
            cholesky_decomposition_enum::LOWER_TRIANGULAR>(lda, C_aad, C, lda, A_aad);
        return true;
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        detail::cholesky_decomposition_aad_impl<float,
            cholesky_decomposition_enum::UPPER_TRIANGULAR>(lda, C_aad, C, lda, A_aad);
        return true;
    default:
        LINALG_THROW("Unsupported enum type!");
    }
}

template <>
bool cholesky_decomposition<double, linalg_backend::NATIVE>::apply_aad(
    double* C_aad, const double* C, linalg_int lda, cholesky_decomposition_enum type, double* A_aad)
{
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        detail::cholesky_decomposition_aad_impl<double,
            cholesky_decomposition_enum::LOWER_TRIANGULAR>(lda, C_aad, C, lda, A_aad);
        return true;
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        detail::cholesky_decomposition_aad_impl<double,
            cholesky_decomposition_enum::UPPER_TRIANGULAR>(lda, C_aad, C, lda, A_aad);
        return true;
    default:
        LINALG_THROW("Unsupported enum type!");
    }
}

}  // namespace linalg
