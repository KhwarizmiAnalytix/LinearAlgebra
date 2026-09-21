#include "include/matrix_operation/linear_solver.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/util/exception.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
// LAPACKE-dependent — see CMakeLists.txt and
// cholesky_decomposition.cxx.
#include <lapacke.h>

#include <vector>
#else
#include <algorithm>

#include "include/common/macros.h"
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// linear_solver() overloads below call — exactly one is ever built.
#if defined(LINALG_ENABLE_MKL)

void solver_mkl_f32(
    float* m, linalg_int* pivot, linalg_int lda, float* x, linear_solver_type type)
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

void solver_mkl_f64(
    double* m, linalg_int* pivot, linalg_int lda, double* x, linear_solver_type type)
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

#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

namespace
{
// See lu_decomposition.cxx: a generic LAPACKE's lapack_int does not
// necessarily match linalg_int, so the pivot buffer (already populated
// with linalg_int-widened values by lu_decomposition's own blas backend)
// is narrowed back down for this call.
std::vector<lapack_int> to_lapack_pivots(const linalg_int* pivot, linalg_int n)
{
    std::vector<lapack_int> ipiv(static_cast<size_t>(n));
    for (linalg_int i = 0; i < n; ++i)
    {
        ipiv[static_cast<size_t>(i)] = static_cast<lapack_int>(pivot[i]);
    }
    return ipiv;
}
}  // namespace

void solver_blas_f32(
    float* m, linalg_int* pivot, linalg_int lda, float* x, linear_solver_type type)
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

void solver_blas_f64(
    double* m, linalg_int* pivot, linalg_int lda, double* x, linear_solver_type type)
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

#else

namespace
{

#define A(i, j) m[(i) * lda + (j)]

template <typename T>
void lu_solve(T* m, LINALG_UNUSED const linalg_int* pivot, linalg_int lda, T* x)
{
#ifdef LINALG_LU_PIVOTING
    for (int i = 0; i < lda; i++)
    {
        std::swap(x[i], x[pivot[i] - 1]);
    }
#endif
    for (linalg_int i = 0; i < lda; i++)
    {
        auto sum = x[i];  //
        for (linalg_int j = 0; j < i; ++j)
        {
            sum -= A(i, j) * x[j];
        }

        x[i] = sum;
    }

    for (linalg_int i = lda - 1; i >= 0; i--)
    {
        auto sum = x[i];  //
        for (linalg_int j = i + 1; j < lda; ++j)
        {
            sum -= A(i, j) * x[j];
        }

        x[i] = sum / A(i, i);
    }
}

template <typename T> void cholesky_solve(T* m, linalg_int lda, T* x)
{
    for (linalg_int i = 0; i < lda; i++)
    {
        auto sum = x[i];
        for (linalg_int j = 0; j < i; ++j)
        {
            sum -= A(i, j) * x[j];
        }

        x[i] = sum / A(i, i);
    }

    for (linalg_int i = lda - 1; i >= 0; i--)
    {
        auto sum = x[i];  //
        for (linalg_int j = i + 1; j < lda; ++j)
        {
            sum -= A(j, i) * x[j];
        }

        x[i] = sum / A(i, i);
    }
}

#undef A

template <typename T>
void linear_solver_scalar_impl(
    T* m, linalg_int* pivot, linalg_int lda, T* x, linear_solver_type type)
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

void solver_scalar_f32(
    float* m, linalg_int* pivot, linalg_int lda, float* x, linear_solver_type type)
{
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

void solver_scalar_f64(
    double* m, linalg_int* pivot, linalg_int lda, double* x, linear_solver_type type)
{
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

#endif

}  // namespace detail

//-----------------------------------------------------------------------------
void linear_solver(
    float* m, linalg_int* pivot, linalg_int lda, float* x, linear_solver_type type)
{
    if (m == nullptr || x == nullptr)
    {
        LINALG_THROW("linear_solver: m and x must not be null");
    }
    if (lda <= 0)
    {
        LINALG_THROW("linear_solver: lda must be positive", lda);
    }
#if defined(LINALG_ENABLE_MKL)
    detail::solver_mkl_f32(m, pivot, lda, x, type);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::solver_blas_f32(m, pivot, lda, x, type);
#else
    detail::solver_scalar_f32(m, pivot, lda, x, type);
#endif
}

//-----------------------------------------------------------------------------
void linear_solver(
    double* m, linalg_int* pivot, linalg_int lda, double* x, linear_solver_type type)
{
    if (m == nullptr || x == nullptr)
    {
        LINALG_THROW("linear_solver: m and x must not be null");
    }
    if (lda <= 0)
    {
        LINALG_THROW("linear_solver: lda must be positive", lda);
    }
#if defined(LINALG_ENABLE_MKL)
    detail::solver_mkl_f64(m, pivot, lda, x, type);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::solver_blas_f64(m, pivot, lda, x, type);
#else
    detail::solver_scalar_f64(m, pivot, lda, x, type);
#endif
}
}  // namespace linalg
