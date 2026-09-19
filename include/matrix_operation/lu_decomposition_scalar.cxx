#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "include/common/configure.h"  // IWYU pragma: keep
#include "include/common/macros.h"
#include "include/matrix_operation/lu_decomposition_dispatch.h"

namespace linalg
{
namespace detail
{
namespace
{

#define A(i, j) m[(i) * lda + (j)]

template <typename T>
bool lu_decomposition_scalar_impl(
    T* m, quarisma_long lda, LINALG_UNUSED quarisma_int* pivot, LINALG_UNUSED T tolerance)
{
    using value_t   = T;
    using size_type = quarisma_long;

#ifdef LINALG_LU_PIVOTING
    // Initialize pivot array
    for (size_type i = 0; i <= lda; ++i)
    {
        pivot[i] = static_cast<quarisma_int>(i + 1);
    }
    quarisma_int& pivotCount = pivot[lda];
#endif
    for (size_type i = 0; i < lda; ++i)
    {
#ifdef LINALG_LU_PIVOTING
        value_t   maxA = std::abs(A(i, i));
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
            for (size_type j = 0; j < lda; ++j)
            {
                auto& ptri = A(i, j);
                auto& ptrj = A(imax, j);
                std::swap(ptri, ptrj);
            }
            pivotCount++;
        }
#endif
        auto norm = 1. / A(i, i);
        auto size = lda - i - 1;

        const auto* const l_i = &A(i, i + 1);
        for (size_type j = i + 1; j < lda; ++j)
        {
            auto* l_j = &A(j, i);
            l_j[0] *= static_cast<T>(norm);
            const auto a_ji = l_j[0];
            l_j++;
            for (size_type k = 0; k < size; ++k)
            {
                l_j[k] -= a_ji * l_i[k];
            }
        }
    }
    return true;
}

#undef A

}  // namespace

bool lu_scalar_f32(float* m, quarisma_int lda, quarisma_int* pivot)
{
    return lu_decomposition_scalar_impl(m, lda, pivot, std::numeric_limits<float>::epsilon());
}

bool lu_scalar_f64(double* m, quarisma_int lda, quarisma_int* pivot)
{
    return lu_decomposition_scalar_impl(m, lda, pivot, std::numeric_limits<double>::epsilon());
}

LINALG_REGISTER_DISPATCH(lu_f32_stub, scalar, lu_scalar_f32);
LINALG_REGISTER_DISPATCH(lu_f64_stub, scalar, lu_scalar_f64);

}  // namespace detail
}  // namespace linalg
