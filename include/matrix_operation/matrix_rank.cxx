#include "include/matrix_operation/matrix_rank.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "include/matrix_operation/svd_decomposition.h"
#include "include/memory/allocator.h"
#include "ThirdParty/Logging/include/logging.h"

namespace linalg
{
namespace detail
{

template <typename T> std::vector<T> singular_values(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    using Allocator       = linalg::allocator<T>;
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

    Allocator::free(VT);
    Allocator::free(U);
    Allocator::free(A_copy);
    return S;
}

template <typename T> linalg_long matrix_rank_impl(const T* A, linalg_long rows, linalg_long columns, linalg_long lda, T tol)
{
    const auto S = singular_values(A, rows, columns, lda);

    T smax = T(0);
    for (const auto& s : S)
    {
        smax = std::max(smax, std::fabs(s));
    }

    const T eff_tol =
        (tol >= T(0)) ? tol : std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(rows, columns)) * smax;

    linalg_long rank = 0;
    for (const auto& s : S)
    {
        if (std::fabs(s) > eff_tol)
        {
            ++rank;
        }
    }
    return rank;
}

template <typename T> T matrix_condition_number_impl(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    const auto S = singular_values(A, rows, columns, lda);

    T smax = T(0);
    T smin = std::numeric_limits<T>::max();
    for (const auto& s : S)
    {
        const T abs_s = std::fabs(s);
        smax          = std::max(smax, abs_s);
        smin          = std::min(smin, abs_s);
    }

    if (smin == T(0))
    {
        return std::numeric_limits<T>::infinity();
    }
    return smax / smin;
}

}  // namespace detail

//-----------------------------------------------------------------------------
linalg_long matrix_rank(const float* A, linalg_long rows, linalg_long columns, linalg_long lda, float tol)
{
    LOGGING_CHECK(A != nullptr, "matrix_rank: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0, "matrix_rank: rows/columns/lda must be positive");
    return detail::matrix_rank_impl(A, rows, columns, lda, tol);
}

//-----------------------------------------------------------------------------
linalg_long matrix_rank(const double* A, linalg_long rows, linalg_long columns, linalg_long lda, double tol)
{
    LOGGING_CHECK(A != nullptr, "matrix_rank: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0, "matrix_rank: rows/columns/lda must be positive");
    return detail::matrix_rank_impl(A, rows, columns, lda, tol);
}

//-----------------------------------------------------------------------------
float matrix_condition_number(const float* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    LOGGING_CHECK(A != nullptr, "matrix_condition_number: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0, "matrix_condition_number: rows/columns/lda must be positive");
    return detail::matrix_condition_number_impl(A, rows, columns, lda);
}

//-----------------------------------------------------------------------------
double matrix_condition_number(const double* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    LOGGING_CHECK(A != nullptr, "matrix_condition_number: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0, "matrix_condition_number: rows/columns/lda must be positive");
    return detail::matrix_condition_number_impl(A, rows, columns, lda);
}

}  // namespace linalg
