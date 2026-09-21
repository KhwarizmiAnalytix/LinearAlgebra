#include "include/matrix_operation/matrix_norm.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "include/matrix_operation/svd_decomposition.h"
#include "include/memory/allocator.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{

template <typename T> T norm_frobenius(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    T sum = T(0);
    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            const T v = A[i * lda + j];
            sum += v * v;
        }
    }
    return std::sqrt(sum);
}

template <typename T> T norm_one(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    T best = T(0);
    for (linalg_long j = 0; j < columns; ++j)
    {
        T col_sum = T(0);
        for (linalg_long i = 0; i < rows; ++i)
        {
            col_sum += std::fabs(A[i * lda + j]);
        }
        best = std::max(best, col_sum);
    }
    return best;
}

template <typename T> T norm_infinity(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    T best = T(0);
    for (linalg_long i = 0; i < rows; ++i)
    {
        T row_sum = T(0);
        for (linalg_long j = 0; j < columns; ++j)
        {
            row_sum += std::fabs(A[i * lda + j]);
        }
        best = std::max(best, row_sum);
    }
    return best;
}

template <typename T> T norm_two(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    using Allocator     = linalg::allocator<T>;
    const linalg_long k = std::min(rows, columns);

    auto* A_copy = Allocator::allocate(rows * columns);
    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            A_copy[i * columns + j] = A[i * lda + j];
        }
    }

    std::vector<T> S(static_cast<std::size_t>(k));
    auto*          U  = Allocator::allocate(rows * k);
    auto*          VT = Allocator::allocate(k * columns);

    linalg::svd_decomposition(rows, columns, A_copy, columns, S.data(), U, k, VT, columns);

    T best = T(0);
    for (linalg_long i = 0; i < k; ++i)
    {
        best = std::max(best, std::fabs(S[static_cast<std::size_t>(i)]));
    }

    Allocator::free(VT);
    Allocator::free(U);
    Allocator::free(A_copy);
    return best;
}

template <typename T> T matrix_norm_impl(const T* A, linalg_long rows, linalg_long columns, linalg_long lda, matrix_norm_type type)
{
    switch (type)
    {
    case matrix_norm_type::FROBENIUS:
        return norm_frobenius(A, rows, columns, lda);
    case matrix_norm_type::ONE:
        return norm_one(A, rows, columns, lda);
    case matrix_norm_type::INFINITY_NORM:
        return norm_infinity(A, rows, columns, lda);
    case matrix_norm_type::TWO:
        return norm_two(A, rows, columns, lda);
    default:
        LINALG_THROW("unsupported matrix_norm_type", static_cast<linalg_int>(type));
    }
}

}  // namespace detail

//-----------------------------------------------------------------------------
float matrix_norm(const float* A, linalg_long rows, linalg_long columns, linalg_long lda, matrix_norm_type type)
{
    if (A == nullptr)
    {
        LINALG_THROW("matrix_norm: A must not be null");
    }
    if (rows == 0 || columns == 0 || lda == 0)
    {
        LINALG_THROW("matrix_norm: rows/columns/lda must be positive");
    }
    return detail::matrix_norm_impl(A, rows, columns, lda, type);
}

//-----------------------------------------------------------------------------
double matrix_norm(const double* A, linalg_long rows, linalg_long columns, linalg_long lda, matrix_norm_type type)
{
    if (A == nullptr)
    {
        LINALG_THROW("matrix_norm: A must not be null");
    }
    if (rows == 0 || columns == 0 || lda == 0)
    {
        LINALG_THROW("matrix_norm: rows/columns/lda must be positive");
    }
    return detail::matrix_norm_impl(A, rows, columns, lda, type);
}

}  // namespace linalg
