#include "include/matrix_operation/cholesky_decomposition.h"

#include "ThirdParty/Logging/include/logging.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
// LAPACKE-dependent (see CMakeLists.txt): Apple's Accelerate
// framework ships CBLAS but not the LAPACKE row-major C wrapper, so this
// branch only compiles when a real lapacke.h was found; otherwise the #else
// below falls through to the scalar implementation at compile time.
#include <lapacke.h>
#else
#include <cmath>
#include <stdexcept>
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// cholesky_decomposition() overloads below call — exactly one is ever built.
#if defined(LINALG_ENABLE_MKL)

bool cholesky_mkl_f32(float* C, linalg_int lda, cholesky_decomposition_enum type)
{
    linalg_int n = lda;
    return LAPACKE_spotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

bool cholesky_mkl_f64(double* C, linalg_int lda, cholesky_decomposition_enum type)
{
    linalg_int n = lda;
    return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

bool cholesky_blas_f32(float* C, linalg_int lda, cholesky_decomposition_enum type)
{
    const auto n = static_cast<int>(lda);
    return LAPACKE_spotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

bool cholesky_blas_f64(double* C, linalg_int lda, cholesky_decomposition_enum type)
{
    const auto n = static_cast<int>(lda);
    return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

#else

namespace
{

enum class matrix_order
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
    else if constexpr (part == cholesky_decomposition_enum::UPPER_TRIANGULAR)
    {
        for (linalg_int i = 0; i < n; ++i)
        {
            for (linalg_int j = 0; j < i; ++j)
            {
                T sum = A[j * lda + i];
                for (linalg_int k = 0; k < j; ++k)
                {
                    sum -= A[k * lda + i] * A[k * lda + j];
                }
                A[j * lda + i] = sum / A[j * lda + j];
            }

            T sum = A[i * lda + i];
            for (linalg_int k = 0; k < i; ++k)
            {
                sum -= A[k * lda + i] * A[k * lda + i];
            }

            if (sum <= 0)
            {
                return i + 1;  // Matrix is not positive-definite
            }

            A[i * lda + i] = std::sqrt(sum);
        }
    }
    else
    {
        throw std::runtime_error("Invalid MatrixPart specified");
    }

    return 0;  // Success
}

}  // namespace

bool cholesky_scalar_f32(float* C, linalg_int lda, cholesky_decomposition_enum type)
{
    linalg_int n = lda;
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return cholesky_decomposition_scalar_impl<float,
                   cholesky_decomposition_enum::LOWER_TRIANGULAR>(n, C, n) == 0;
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        return cholesky_decomposition_scalar_impl<float,
                   cholesky_decomposition_enum::UPPER_TRIANGULAR>(n, C, n) == 0;
    default:
        LOGGING_THROW("Unsupported enum type!");
    }
}

bool cholesky_scalar_f64(double* C, linalg_int lda, cholesky_decomposition_enum type)
{
    linalg_int n = lda;
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return cholesky_decomposition_scalar_impl<double,
                   cholesky_decomposition_enum::LOWER_TRIANGULAR>(n, C, n) == 0;
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        return cholesky_decomposition_scalar_impl<double,
                   cholesky_decomposition_enum::UPPER_TRIANGULAR>(n, C, n) == 0;
    default:
        LOGGING_THROW("Unsupported enum type!");
    }
}

#endif

// Reverse-mode adjoint of cholesky_decomposition: no vendor library (LAPACK/
// cuSOLVER) exposes this, so it always runs the scalar implementation
// directly, regardless of which backend cholesky_decomposition itself used.
enum class aad_matrix_order
{
    row_major,
    col_major
};

template <typename T,
    cholesky_decomposition_enum part,
    aad_matrix_order            order = aad_matrix_order::row_major>
bool cholesky_decomposition_aad_impl(
    linalg_int n, T* C_aad, const T* C, linalg_long lda, T* A_aad)
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

            auto* u_i_aad = &C_aad[i * lda];
            auto* a_i_aad = &A_aad[i * lda];
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

    return true;  // Success
}

}  // namespace detail

//-----------------------------------------------------------------------------
bool cholesky_decomposition(float* C, linalg_int lda, linalg::cholesky_decomposition_enum type)
{
    LOGGING_CHECK(C != nullptr, "cholesky_decomposition: C must not be null");
    LOGGING_CHECK(lda > 0, "cholesky_decomposition: lda must be positive", lda);
#if defined(LINALG_ENABLE_MKL)
    return detail::cholesky_mkl_f32(C, lda, type);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::cholesky_blas_f32(C, lda, type);
#else
    return detail::cholesky_scalar_f32(C, lda, type);
#endif
}

//-----------------------------------------------------------------------------
bool cholesky_decomposition(double* C, linalg_int lda, linalg::cholesky_decomposition_enum type)
{
    LOGGING_CHECK(C != nullptr, "cholesky_decomposition: C must not be null");
    LOGGING_CHECK(lda > 0, "cholesky_decomposition: lda must be positive", lda);
#if defined(LINALG_ENABLE_MKL)
    return detail::cholesky_mkl_f64(C, lda, type);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::cholesky_blas_f64(C, lda, type);
#else
    return detail::cholesky_scalar_f64(C, lda, type);
#endif
}

//-----------------------------------------------------------------------------
bool cholesky_decomposition_aad(float*  C_aad,
    const float*                        C,
    linalg_int                        lda,
    linalg::cholesky_decomposition_enum type,
    float*                              A_aad)
{
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return detail::cholesky_decomposition_aad_impl<float,
            cholesky_decomposition_enum::LOWER_TRIANGULAR>(lda, C_aad, C, lda, A_aad);
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        return detail::cholesky_decomposition_aad_impl<float,
            cholesky_decomposition_enum::UPPER_TRIANGULAR>(lda, C_aad, C, lda, A_aad);
    default:
        LOGGING_THROW("Unsupported enum type!");
    }
}

//-----------------------------------------------------------------------------
bool cholesky_decomposition_aad(double* C_aad,
    const double*                       C,
    linalg_int                        lda,
    linalg::cholesky_decomposition_enum type,
    double*                             A_aad)
{
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return detail::cholesky_decomposition_aad_impl<double,
            cholesky_decomposition_enum::LOWER_TRIANGULAR>(lda, C_aad, C, lda, A_aad);
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        return detail::cholesky_decomposition_aad_impl<double,
            cholesky_decomposition_enum::UPPER_TRIANGULAR>(lda, C_aad, C, lda, A_aad);
    default:
        LOGGING_THROW("Unsupported enum type!");
    }
}
}  // namespace linalg
