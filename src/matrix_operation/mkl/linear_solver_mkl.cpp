#if LINALG_HAS_MKL

#include "include/matrix_operation/linear_solver.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include <include/logging.h>

#include <mkl.h>

// MKL specializations for linear_solver.h (the native backend lives in
// linear_solver.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
// on; the #if below is kept as a second guard.
namespace linalg
{

// Same checks as linear_solver.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_solver_inputs(const void* m, const void* x, linalg_int lda)
{
    LOGGING_CHECK(m != nullptr && x != nullptr, "linear_solver::apply: m and x must not be null");
    LOGGING_CHECK(lda > 0, "linear_solver::apply: lda must be positive", lda);
}

template <>
void linear_solver<float, linalg_backend::mkl>::apply(float* m,
    linalg_int*                                               pivot,
    linalg_int                                                lda,
    float*                                                    x,
    linear_solver_type                                        type,
    void*                                                     workspace,
    size_t                                                    workspace_size)
{
    check_solver_inputs(m, x, lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        if (lu_decomposition<float, kActiveBackend>::apply(m, lda, pivot, workspace, workspace_size))
        {
            LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
        }
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition<float, kActiveBackend>::apply(
                m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
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

template <>
void linear_solver<double, linalg_backend::mkl>::apply(double* m,
    linalg_int*                                                 pivot,
    linalg_int                                                  lda,
    double*                                                     x,
    linear_solver_type                                          type,
    void*                                                       workspace,
    size_t                                                      workspace_size)
{
    check_solver_inputs(m, x, lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        if (lu_decomposition<double, kActiveBackend>::apply(m, lda, pivot, workspace, workspace_size))
        {
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
        }
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition<double, kActiveBackend>::apply(
                m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
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

template <>
size_t linear_solver<float, linalg_backend::mkl>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        return lu_decomposition<float, kActiveBackend>::workspace_size(lda);
    default:
        return 0;
    }
}

template <>
size_t linear_solver<double, linalg_backend::mkl>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        return lu_decomposition<double, kActiveBackend>::workspace_size(lda);
    default:
        return 0;
    }
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
