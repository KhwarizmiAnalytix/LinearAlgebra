#include "include/matrix_operation/pseudo_inverse.h"

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

template <typename T>
void pseudo_inverse_impl(
    linalg_long rows, linalg_long columns, const T* A, linalg_long lda, T* Ainv, linalg_long ldai, T tol)
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

    T smax = T(0);
    for (linalg_long i = 0; i < k; ++i)
    {
        smax = std::max(smax, std::fabs(S[static_cast<std::size_t>(i)]));
    }

    const T eff_tol = (tol >= T(0))
        ? tol
        : std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(rows, columns)) * smax;

    for (linalg_long c = 0; c < columns; ++c)
    {
        for (linalg_long r = 0; r < rows; ++r)
        {
            T sum = T(0);
            for (linalg_long i = 0; i < k; ++i)
            {
                const T s = S[static_cast<std::size_t>(i)];
                if (s > eff_tol)
                {
                    sum += VT[i * columns + c] * (T(1) / s) * U[r * k + i];
                }
            }
            Ainv[c * ldai + r] = sum;
        }
    }

    Allocator::free(VT);
    Allocator::free(U);
    Allocator::free(A_copy);
}

}  // namespace detail

//-----------------------------------------------------------------------------
void pseudo_inverse(
    linalg_long rows, linalg_long columns, const float* A, linalg_long lda, float* Ainv, linalg_long ldai, float tol)
{
    LOGGING_CHECK(A != nullptr && Ainv != nullptr, "pseudo_inverse: A and Ainv must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0, "pseudo_inverse: rows/columns/lda/ldai must be positive");
    detail::pseudo_inverse_impl(rows, columns, A, lda, Ainv, ldai, tol);
}

//-----------------------------------------------------------------------------
void pseudo_inverse(
    linalg_long rows, linalg_long columns, const double* A, linalg_long lda, double* Ainv, linalg_long ldai, double tol)
{
    LOGGING_CHECK(A != nullptr && Ainv != nullptr, "pseudo_inverse: A and Ainv must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0, "pseudo_inverse: rows/columns/lda/ldai must be positive");
    detail::pseudo_inverse_impl(rows, columns, A, lda, Ainv, ldai, tol);
}

}  // namespace linalg
