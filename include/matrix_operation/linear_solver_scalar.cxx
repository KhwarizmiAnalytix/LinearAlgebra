#include <algorithm>

#include "include/common/macros.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver_dispatch.h"
#include "include/matrix_operation/lu_decomposition.h"

namespace linalg
{
namespace detail
{
namespace
{

#define A(i, j) m[(i) * lda + (j)]

template <typename T>
void lu_solve(T* m, LINALG_UNUSED const quarisma_int* pivot, quarisma_int lda, T* x)
{
#ifdef LINALG_LU_PIVOTING
    for (int i = 0; i < lda; i++)
    {
        std::swap(x[i], x[pivot[i] - 1]);
    }
#endif
    for (quarisma_int i = 0; i < lda; i++)
    {
        auto sum = x[i];  //
        for (quarisma_int j = 0; j < i; ++j)
        {
            sum -= A(i, j) * x[j];
        }

        x[i] = sum;
    }

    for (quarisma_int i = lda - 1; i >= 0; i--)
    {
        auto sum = x[i];  //
        for (quarisma_int j = i + 1; j < lda; ++j)
        {
            sum -= A(i, j) * x[j];
        }

        x[i] = sum / A(i, i);
    }
}

template <typename T>
void cholesky_solve(T* m, quarisma_int lda, T* x)
{
    for (quarisma_int i = 0; i < lda; i++)
    {
        auto sum = x[i];
        for (quarisma_int j = 0; j < i; ++j)
        {
            sum -= A(i, j) * x[j];
        }

        x[i] = sum / A(i, i);
    }

    for (quarisma_int i = lda - 1; i >= 0; i--)
    {
        auto sum = x[i];  //
        for (quarisma_int j = i + 1; j < lda; ++j)
        {
            sum -= A(j, i) * x[j];
        }

        x[i] = sum / A(i, i);
    }
}

#undef A

template <typename T>
void linear_solver_scalar_impl(T* m, quarisma_int* pivot, quarisma_int lda, T* x, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        {
            if (lu_decomposition(m, lda, pivot))
            {
                lu_solve(m, pivot, lda, x);
            }
        }
        break;

        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            lu_solve(m, pivot, lda, x);
            break;

        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        {
            if (cholesky_decomposition(m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                cholesky_solve(m, lda, x);
            }
        }
        break;

        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            cholesky_solve(m, lda, x);
            break;
    }
}

}  // namespace

void solver_scalar_f32(float* m, quarisma_int* pivot, quarisma_int lda, float* x, linear_solver_type type)
{
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

void solver_scalar_f64(
    double* m, quarisma_int* pivot, quarisma_int lda, double* x, linear_solver_type type)
{
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

LINALG_REGISTER_DISPATCH(solver_f32_stub, scalar, solver_scalar_f32);
LINALG_REGISTER_DISPATCH(solver_f64_stub, scalar, solver_scalar_f64);

}  // namespace detail
}  // namespace linalg
