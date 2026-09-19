#include "include/common/configure.h"  // IWYU pragma: keep

// LAPACKE-dependent — see include/common/configure.h.in and
// cholesky_decomposition_blas.cxx.
#if defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

#include <lapacke.h>

#include <vector>

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

// See lu_decomposition_blas.cxx.
std::vector<lapack_int> to_lapack_pivots(const quarisma_int* pivot, quarisma_int n)
{
    std::vector<lapack_int> ipiv(static_cast<size_t>(n));
    for (quarisma_int i = 0; i < n; ++i)
    {
        ipiv[static_cast<size_t>(i)] = static_cast<lapack_int>(pivot[i]);
    }
    return ipiv;
}

void invert_blas_impl_f32(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    const auto n = static_cast<lapack_int>(lda);
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                auto ipiv = to_lapack_pivots(pivot, lda);
                LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, m, n, ipiv.data());
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        {
            auto ipiv = to_lapack_pivots(pivot, lda);
            LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, m, n, ipiv.data());
            break;
        }
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

void invert_blas_impl_f64(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    const auto n = static_cast<lapack_int>(lda);
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                auto ipiv = to_lapack_pivots(pivot, lda);
                LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, m, n, ipiv.data());
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        {
            auto ipiv = to_lapack_pivots(pivot, lda);
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, m, n, ipiv.data());
            break;
        }
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

}  // namespace

void invert_blas_f32(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    invert_blas_impl_f32(m, pivot, lda, type);
}

void invert_blas_f64(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    invert_blas_impl_f64(m, pivot, lda, type);
}

LINALG_REGISTER_DISPATCH(invert_f32_stub, blas, invert_blas_f32);
LINALG_REGISTER_DISPATCH(invert_f64_stub, blas, invert_blas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_BLAS && LINALG_BLAS_HAS_LAPACKE
