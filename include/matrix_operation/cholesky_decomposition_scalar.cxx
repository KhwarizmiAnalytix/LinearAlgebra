#include <cmath>
#include <stdexcept>

#include "include/matrix_operation/cholesky_decomposition_dispatch.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{
namespace
{

enum class matrix_order
{
    row_major,
    col_major
};

template <typename T, cholesky_decomposition_enum part, matrix_order order = matrix_order::row_major>
quarisma_int cholesky_decomposition_scalar_impl(quarisma_int n, T* A, quarisma_int lda)
{
    if constexpr (order == matrix_order::col_major)
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

}  // namespace

bool cholesky_scalar_f32(float* C, quarisma_int lda, cholesky_decomposition_enum type)
{
    quarisma_int n = lda;
    switch (type)
    {
        case cholesky_decomposition_enum::LOWER_TRIANGULAR:
            return cholesky_decomposition_scalar_impl<
                       float, cholesky_decomposition_enum::LOWER_TRIANGULAR>(n, C, n) == 0;
        case cholesky_decomposition_enum::UPPER_TRIANGULAR:
            return cholesky_decomposition_scalar_impl<
                       float, cholesky_decomposition_enum::UPPER_TRIANGULAR>(n, C, n) == 0;
        default:
            LINALG_THROW("Unsupported enum type!");
    }
}

bool cholesky_scalar_f64(double* C, quarisma_int lda, cholesky_decomposition_enum type)
{
    quarisma_int n = lda;
    switch (type)
    {
        case cholesky_decomposition_enum::LOWER_TRIANGULAR:
            return cholesky_decomposition_scalar_impl<
                       double, cholesky_decomposition_enum::LOWER_TRIANGULAR>(n, C, n) == 0;
        case cholesky_decomposition_enum::UPPER_TRIANGULAR:
            return cholesky_decomposition_scalar_impl<
                       double, cholesky_decomposition_enum::UPPER_TRIANGULAR>(n, C, n) == 0;
        default:
            LINALG_THROW("Unsupported enum type!");
    }
}

LINALG_REGISTER_DISPATCH(cholesky_f32_stub, scalar, cholesky_scalar_f32);
LINALG_REGISTER_DISPATCH(cholesky_f64_stub, scalar, cholesky_scalar_f64);

}  // namespace detail
}  // namespace linalg
