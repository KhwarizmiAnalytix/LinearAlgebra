#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_MKL

#include <mkl.h>

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

void invert_mkl_impl_f32(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                LAPACKE_sgetri(LAPACK_ROW_MAJOR, lda, m, lda, pivot);
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            LAPACKE_sgetri(LAPACK_ROW_MAJOR, lda, m, lda, pivot);
            break;
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

void invert_mkl_impl_f64(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                LAPACKE_dgetri(LAPACK_ROW_MAJOR, lda, m, lda, pivot);
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, lda, m, lda, pivot);
            break;
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

}  // namespace

void invert_mkl_f32(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    invert_mkl_impl_f32(m, pivot, lda, type);
}

void invert_mkl_f64(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    invert_mkl_impl_f64(m, pivot, lda, type);
}

LINALG_REGISTER_DISPATCH(invert_f32_stub, mkl, invert_mkl_f32);
LINALG_REGISTER_DISPATCH(invert_f64_stub, mkl, invert_mkl_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_MKL
