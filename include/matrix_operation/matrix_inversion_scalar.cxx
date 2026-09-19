#include <algorithm>
#include <vector>

#include "include/common/macros.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/matrix_inversion_dispatch.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{
namespace
{

#define A(i, j) m[(i) * lda + (j)]
#define IA(i, j) inv_m[(i) * lda + (j)]

template <typename T>
void lu_invert(T* m, LINALG_UNUSED const quarisma_int* pivot, quarisma_int lda)
{
    using size_type = quarisma_int;
    using value_t   = T;
    std::vector<double> inv_m(lda * lda);
    std::vector<double> work(lda);
    for (size_type j = 0; j < lda; ++j)
    {
        for (size_type i = 0; i < lda; ++i)
        {
            value_t sum = i == j ? 1.0 : 0.0;

            for (size_type k = 0; k < i; ++k)
            {
                sum -= A(i, k) * work[k];
            }

            work[i] = static_cast<value_t>(sum);
        }

        for (size_type i = lda - 1; i >= 0; --i)
        {
            auto& sum = work[i];

            for (size_type k = i + 1; k < lda; ++k)
            {
                sum -= A(i, k) * work[k];
            }

            sum /= A(i, i);
        }

        for (size_type i = 0; i < lda; ++i)
        {
            IA(i, j) = work[i];
        }
    }
    for (size_type j = 0; j < lda; ++j)
    {
        for (size_type i = 0; i < lda; ++i)
        {
            A(i, j) = IA(i, j);
        }
    }
#ifdef LINALG_LU_PIVOTING
    for (size_type i = lda - 1; i >= 0; --i)
    {
        size_type p = pivot[i] - 1;
        if (p != i)
        {
            for (size_type j = 0; j < lda; ++j)
            {
                auto& a_ij = A(j, i);
                auto& a_pj = A(j, p);
                std::swap(a_ij, a_pj);
            }
        }
    }
#endif
}

template <typename T>
void cholesky_invert(T* m, quarisma_int lda)
{
    using value_t   = T;
    using size_type = quarisma_int;

    std::vector<T> work(lda);

    for (size_type j = 0; j < lda; ++j)
    {
        // Compute L^-1
        for (size_type i = j; i < lda; ++i)
        {
            auto sum = static_cast<value_t>((i == j) ? 1.0 : 0.0);
            for (size_type k = j; k < i; ++k)
            {
                sum -= A(i, k) * A(k, j);
            }
            A(i, j) = sum / A(i, i);
        }

        // Compute (L^T)^-1 * L^-1 for the j-th column
        for (size_type i = lda - 1; i >= 0; --i)
        {
            auto sum = static_cast<value_t>(0.0);
            for (size_type k = i; k < lda; ++k)
            {
                sum += A(k, i) * A(k, j);
            }
            work[i] = sum;
        }

        // Copy the result back to the original matrix
        for (size_type i = 0; i <= j; ++i)
        {
            A(i, j) = work[i];
        }
    }

    // Fill in the lower triangular part
    for (size_type i = 1; i < lda; ++i)
    {
        for (size_type j = 0; j < i; ++j)
        {
            A(i, j) = A(j, i);
        }
    }
}

#undef A
#undef IA

template <typename T>
void matrix_invert_scalar_impl(T* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        {
            if (lu_decomposition(m, lda, pivot))
            {
                lu_invert(m, pivot, lda);
            }
        }
        break;

        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            lu_invert(m, pivot, lda);
            break;

        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        {
            if (cholesky_decomposition(m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                cholesky_invert(m, lda);
            }
        }
        break;

        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            cholesky_invert(m, lda);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

}  // namespace

void invert_scalar_f32(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    matrix_invert_scalar_impl(m, pivot, lda, type);
}

void invert_scalar_f64(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    matrix_invert_scalar_impl(m, pivot, lda, type);
}

LINALG_REGISTER_DISPATCH(invert_f32_stub, scalar, invert_scalar_f32);
LINALG_REGISTER_DISPATCH(invert_f64_stub, scalar, invert_scalar_f64);

}  // namespace detail
}  // namespace linalg
