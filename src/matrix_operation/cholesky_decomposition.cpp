#include "include/matrix_operation/cholesky_decomposition.h"

#include <cstdint>

#include "include/common/backend.h"
#include "include/common/linalg_checks.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

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

enum class matrix_order : std::uint8_t
{
    row_major,
    col_major
};

template <typename T,
    cholesky_decomposition_enum part,
    matrix_order                order = matrix_order::row_major>
linalg_int cholesky_decomposition_scalar_impl(linalg_int n, T* A, linalg_int lda)
{
    if constexpr (order == matrix_order::col_major)
    {
        throw std::runtime_error("Column-major order is not supported in this implementation.");
    }

    if constexpr (part == cholesky_decomposition_enum::LOWER_TRIANGULAR)  // NOLINT
    {
        for (linalg_int i = 0; i < n; ++i)
        {
            auto* a_i = &A[i * lda];

            for (linalg_int j = 0; j < i; ++j)
            {
                auto* a_j = &A[j * lda];
                T     sum = a_i[j];
                for (linalg_int k = 0; k < j; ++k)
                {
                    sum -= a_i[k] * a_j[k];
                }
                a_i[j] = sum / a_j[j];
                a_j[i] = 0.;
            }

            T sum = a_i[i];
            for (linalg_int k = 0; k < i; ++k)
            {
                sum -= a_i[k] * a_i[k];
            }

            if (sum <= 0)
            {
                return i + 1;  // Matrix is not positive-definite
            }

            a_i[i] = std::sqrt(sum);
        }
    }
    else
    {
        throw std::runtime_error("Invalid MatrixPart specified");
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

// Dot product of two length-`len` contiguous rows, used below for the panel solve's
// forward-substitution reduction and the trailing submatrix's rank update -- the two
// inner loops blocking turns into fixed-shape reductions worth vectorizing explicitly.
// Uses linalg::packet<T> (src/common/packet.h), the same Eigen-derived SIMD abstraction
// multiplication.cpp's native GEMM kernel is built on, instead of hoping the compiler
// auto-vectorizes a scalar loop.
template <typename T>
inline T cholesky_dot(const T* x, const T* y, linalg_int len)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;

    typename P::packet_t acc;
    P::set(static_cast<T>(0), acc);

    linalg_int c = 0;
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

template <typename T>
linalg_int cholesky_decomposition_blocked_lower(linalg_int n, T* A, linalg_int lda)
{
    for (linalg_int k = 0; k < n; k += kCholeskyBlockSize)
    {
        const linalg_int bs  = std::min(kCholeskyBlockSize, n - k);
        T* const         l11 = &A[k * lda + k];

        const linalg_int rc = cholesky_decomposition_scalar_impl<T,
            cholesky_decomposition_enum::LOWER_TRIANGULAR>(bs, l11, lda);
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
bool cholesky_decomposition<float, linalg_backend::native>::apply(
    float* C, linalg_int lda, cholesky_decomposition_enum type)
{
    detail::check_cholesky_inputs(C, lda);
    return cholesky_decomposition_apply_impl<float>(C, lda, type);
}

template <>
bool cholesky_decomposition<double, linalg_backend::native>::apply(
    double* C, linalg_int lda, cholesky_decomposition_enum type)
{
    detail::check_cholesky_inputs(C, lda);
    return cholesky_decomposition_apply_impl<double>(C, lda, type);
}

// Reverse-mode adjoint of cholesky_decomposition<T, Backend>::apply_aad: no vendor
// library (LAPACK/cuSOLVER) exposes this, so it always runs the scalar
// implementation directly, regardless of which backend is active.
namespace detail
{
enum class aad_matrix_order : std::uint8_t
{
    row_major,
    col_major
};

template <typename T,
    cholesky_decomposition_enum part,
    aad_matrix_order            order = aad_matrix_order::row_major>
void cholesky_decomposition_aad_impl(linalg_int n, T* C_aad, const T* C, linalg_long lda, T* A_aad)
{
    if constexpr (order == aad_matrix_order::col_major)
    {
        throw std::runtime_error("Column-major order is not supported in this implementation.");
    }

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
bool cholesky_decomposition<float, linalg_backend::native>::apply_aad(
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
bool cholesky_decomposition<double, linalg_backend::native>::apply_aad(
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
