#include "linear_algebra/matrix_operation/cholesky_decomposition.h"

#include <cmath>

#include "linear_algebra/util/exception.h"

#ifdef LINALG_ENABLE_MKL
#include <mkl.h>

#include <algorithm>
#endif

namespace linalg
{
enum class MatrixOrder
{
    RowMajor,
    ColMajor
};

template <typename T, cholesky_decomposition_enum part, MatrixOrder order = MatrixOrder::RowMajor>
quarisma_int cholesky_decomposition(quarisma_int n, T* A, quarisma_int lda)
{
    if constexpr (order == MatrixOrder::ColMajor)
    {
        throw std::runtime_error("Column-major order is not supported in this implementation.");
    }

    if constexpr (part == cholesky_decomposition_enum::LOWER_TRIANGULAR)  //NOLINT
    {
        for (quarisma_int i = 0; i < n; ++i)
        {
            auto* a_i = &A[i * lda];

            for (quarisma_int j = 0; j < i; ++j)
            {
                auto* a_j = &A[j * lda];
                T     sum = a_i[j];
                for (quarisma_int k = 0; k < j; ++k)
                {
                    sum -= a_i[k] * a_j[k];
                }
                a_i[j] = sum / a_j[j];
                a_j[i] = 0.;
            }

            T sum = a_i[i];
            for (quarisma_int k = 0; k < i; ++k)
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
        for (quarisma_int i = 0; i < n; ++i)
        {
            for (quarisma_int j = 0; j < i; ++j)
            {
                T sum = A[j * lda + i];
                for (quarisma_int k = 0; k < j; ++k)
                {
                    sum -= A[k * lda + i] * A[k * lda + j];
                }
                A[j * lda + i] = sum / A[j * lda + j];
            }

            T sum = A[i * lda + i];
            for (quarisma_int k = 0; k < i; ++k)
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

template <typename T, cholesky_decomposition_enum part, MatrixOrder order = MatrixOrder::RowMajor>
bool cholesky_decomposition_aad(quarisma_int n, T* C_aad, const T* C, quarisma_long lda, T* A_aad)
{
    if constexpr (order == MatrixOrder::ColMajor)
    {
        throw std::runtime_error("Column-major order is not supported in this implementation.");
    }

    if constexpr (part == cholesky_decomposition_enum::LOWER_TRIANGULAR)  //NOLINT
    {
        for (quarisma_int i = n - 1; i >= 0; --i)
        {
            const auto* const l_i = &C[i * lda];

            auto* l_i_aad = &C_aad[i * lda];
            auto* a_i_aad = &A_aad[i * lda];
            {
                // AAD: l_i[i] = std::sqrt(sum);
                auto sum_aad = 0.5 * l_i_aad[i] / l_i[i];
                // l_i_aad[i]   = 0.;

                for (quarisma_int k = 0; k < i; ++k)
                {
                    // AAD: sum -= l_i[k] * l_i[k];
                    l_i_aad[k] -= static_cast<T>(2. * sum_aad * l_i[k]);
                }

                // AAD: auto sum = a_i[i];
                a_i_aad[i] += static_cast<T>(sum_aad);
            }
            for (quarisma_int j = i - 1; j >= 0; --j)
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

                    for (quarisma_int k = 0; k < j; ++k)
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
        for (quarisma_int i = n - 1; i >= 0; --i)
        {
            const auto* const u_i = &C[i * lda];

            auto* u_i_aad = &C_aad[i * lda];
            auto* a_i_aad = &A_aad[i * lda];
            {
                // AAD: u_i[i] = std::sqrt(sum);
                auto sum_aad = 0.5 * u_i_aad[i] / u_i[i];
                // u_i_aad[i]   = 0.;

                for (quarisma_int k = 0; k < i; ++k)
                {
                    // AAD: sum -=  C[k * lda + i] *  C[k * lda + i];
                    C_aad[k * lda + i] -= static_cast<T>(2. * sum_aad * C[k * lda + i]);
                }

                // AAD: auto sum = a_i[i];
                a_i_aad[i] += static_cast<T>(sum_aad);
            }
            for (quarisma_int j = i - 1; j >= 0; --j)
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

                    for (quarisma_int k = 0; k < j; ++k)
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

//-----------------------------------------------------------------------------
bool cholesky_decomposition(float* C, quarisma_int lda, linalg::cholesky_decomposition_enum type)
{
    quarisma_int n = lda;

#ifdef LINALG_ENABLE_MKL

    auto info = (LAPACKE_spotrf(LAPACK_ROW_MAJOR, (char)type, n, C, n) == 0);
    return info;

#else
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return cholesky_decomposition<float, cholesky_decomposition_enum::LOWER_TRIANGULAR>(
                   n, C, n) == 0;
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        return cholesky_decomposition<float, cholesky_decomposition_enum::UPPER_TRIANGULAR>(
                   n, C, n) == 0;
    default:
        LINALG_THROW("Unsupported enum type!");
    }
#endif
}

//-----------------------------------------------------------------------------
bool cholesky_decomposition_aad(
    float*                              C_aad,
    const float*                        C,
    quarisma_int                          lda,
    linalg::cholesky_decomposition_enum type,
    float*                              A_aad)
{
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return cholesky_decomposition_aad<float, cholesky_decomposition_enum::LOWER_TRIANGULAR>(
            lda, C_aad, C, lda, A_aad);
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        return cholesky_decomposition_aad<float, cholesky_decomposition_enum::UPPER_TRIANGULAR>(
            lda, C_aad, C, lda, A_aad);
    default:
        LINALG_THROW("Unsupported enum type!");
    }
}

//-----------------------------------------------------------------------------
bool cholesky_decomposition(double* C, quarisma_int lda, linalg::cholesky_decomposition_enum type)
{
    auto n = static_cast<quarisma_int>(lda);
#ifdef LINALG_ENABLE_MKL

    bool info = (LAPACKE_dpotrf(LAPACK_ROW_MAJOR, (char)type, n, C, n) == 0);
    return info;

#else
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return cholesky_decomposition<double, cholesky_decomposition_enum::LOWER_TRIANGULAR>(
                   n, C, n) == 0;
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        return cholesky_decomposition<double, cholesky_decomposition_enum::UPPER_TRIANGULAR>(
                   n, C, n) == 0;
    default:
        LINALG_THROW("Unsupported enum type!");
    }
#endif
}

//-----------------------------------------------------------------------------
bool cholesky_decomposition_aad(
    double*                             C_aad,
    const double*                       C,
    quarisma_int                          lda,
    linalg::cholesky_decomposition_enum type,
    double*                             A_aad)
{
    switch (type)
    {
    case cholesky_decomposition_enum::LOWER_TRIANGULAR:
        return cholesky_decomposition_aad<double, cholesky_decomposition_enum::LOWER_TRIANGULAR>(
            lda, C_aad, C, lda, A_aad);
    case cholesky_decomposition_enum::UPPER_TRIANGULAR:
        return cholesky_decomposition_aad<double, cholesky_decomposition_enum::UPPER_TRIANGULAR>(
            lda, C_aad, C, lda, A_aad);
    default:
        LINALG_THROW("Unsupported enum type!");
    }
}
}  // namespace linalg
