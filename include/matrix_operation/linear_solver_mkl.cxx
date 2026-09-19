#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_MKL

#include <mkl.h>

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver_dispatch.h"
#include "include/matrix_operation/lu_decomposition.h"

namespace linalg
{
namespace detail
{
namespace
{

void solver_mkl_impl_f32(
    float* m, quarisma_int* pivot, quarisma_int lda, float* x, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
            break;
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
            break;
    }
}

void solver_mkl_impl_f64(
    double* m, quarisma_int* pivot, quarisma_int lda, double* x, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
            break;
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
            break;
    }
}

}  // namespace

void solver_mkl_f32(float* m, quarisma_int* pivot, quarisma_int lda, float* x, linear_solver_type type)
{
    solver_mkl_impl_f32(m, pivot, lda, x, type);
}

void solver_mkl_f64(double* m, quarisma_int* pivot, quarisma_int lda, double* x, linear_solver_type type)
{
    solver_mkl_impl_f64(m, pivot, lda, x, type);
}

LINALG_REGISTER_DISPATCH(solver_f32_stub, mkl, solver_mkl_f32);
LINALG_REGISTER_DISPATCH(solver_f64_stub, mkl, solver_mkl_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_MKL
