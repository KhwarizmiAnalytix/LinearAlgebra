#include "include/common/configure.h"  // IWYU pragma: keep

// LAPACKE-dependent — see include/common/configure.h.in and
// cholesky_decomposition_blas.cxx.
#if defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

#include <lapacke.h>

#include <vector>

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver_dispatch.h"
#include "include/matrix_operation/lu_decomposition.h"

namespace linalg
{
namespace detail
{
namespace
{

// See lu_decomposition_blas.cxx: a generic LAPACKE's lapack_int does not
// necessarily match quarisma_int, so the pivot buffer (already populated
// with quarisma_int-widened values by lu_decomposition's own blas backend)
// is narrowed back down for this call.
std::vector<lapack_int> to_lapack_pivots(const quarisma_int* pivot, quarisma_int n)
{
    std::vector<lapack_int> ipiv(static_cast<size_t>(n));
    for (quarisma_int i = 0; i < n; ++i)
    {
        ipiv[static_cast<size_t>(i)] = static_cast<lapack_int>(pivot[i]);
    }
    return ipiv;
}

void solver_blas_impl_f32(
    float* m, quarisma_int* pivot, quarisma_int lda, float* x, linear_solver_type type)
{
    const auto n = static_cast<lapack_int>(lda);
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                auto ipiv = to_lapack_pivots(pivot, lda);
                LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv.data(), x, 1);
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', n, 1, m, n, x, 1);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        {
            auto ipiv = to_lapack_pivots(pivot, lda);
            LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv.data(), x, 1);
            break;
        }
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', n, 1, m, n, x, 1);
            break;
    }
}

void solver_blas_impl_f64(
    double* m, quarisma_int* pivot, quarisma_int lda, double* x, linear_solver_type type)
{
    const auto n = static_cast<lapack_int>(lda);
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                auto ipiv = to_lapack_pivots(pivot, lda);
                LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv.data(), x, 1);
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, 1, m, n, x, 1);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        {
            auto ipiv = to_lapack_pivots(pivot, lda);
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv.data(), x, 1);
            break;
        }
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, 1, m, n, x, 1);
            break;
    }
}

}  // namespace

void solver_blas_f32(float* m, quarisma_int* pivot, quarisma_int lda, float* x, linear_solver_type type)
{
    solver_blas_impl_f32(m, pivot, lda, x, type);
}

void solver_blas_f64(double* m, quarisma_int* pivot, quarisma_int lda, double* x, linear_solver_type type)
{
    solver_blas_impl_f64(m, pivot, lda, x, type);
}

LINALG_REGISTER_DISPATCH(solver_f32_stub, blas, solver_blas_f32);
LINALG_REGISTER_DISPATCH(solver_f64_stub, blas, solver_blas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_BLAS && LINALG_BLAS_HAS_LAPACKE
