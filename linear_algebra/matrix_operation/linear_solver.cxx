#include "linear_algebra/matrix_operation/linear_solver.h"

#include <algorithm>

#include "linear_algebra/common/macros.h"
#include "linear_algebra/matrix_operation/cholesky_decomposition.h"
#include "linear_algebra/matrix_operation/lu_decomposition.h"
#include "linear_algebra/memory/allocator.h"
#include "linear_algebra/util/exception.h"

#ifdef LINALG_ENABLE_MKL
#include <mkl.h>

#include <type_traits>
#endif

namespace linalg
{
namespace
{
//-----------------------------------------------------------------------------
#define A(i, j) m[(i) * lda + (j)]
#define IA(i, j) inv_m[(i) * lda + (j)]

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
#undef A
#undef IA

//-----------------------------------------------------------------------------
template <typename T>
void linear_solver_helper(T* m, quarisma_int* pivot, quarisma_int lda, T* x, linear_solver_type type)
{
#ifdef LINALG_ENABLE_MKL

    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        if (lu_decomposition(m, lda, pivot))
        {
            if constexpr (std::is_same_v<T, float>)
            {
                LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
            }
            if constexpr (std::is_same_v<T, double>)
            {
                LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
            }
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    {
        if (cholesky_decomposition(m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            if constexpr (std::is_same_v<T, float>)
            {
                LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
            }
            if constexpr (std::is_same_v<T, double>)
            {
                LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
            }
        }

        break;
    }
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        if constexpr (std::is_same_v<T, float>)
        {
            LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
        }
        if constexpr (std::is_same_v<T, double>)
        {
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
        }

        break;
    }
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
    {
        if constexpr (std::is_same_v<T, float>)
        {
            LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
        }
        if constexpr (std::is_same_v<T, double>)
        {
            LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
        }
        break;
    }
    }
#else
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
#endif
}
}  // namespace

//-----------------------------------------------------------------------------
void linear_solver(float* m, quarisma_int* pivot, quarisma_int lda, float* x, linear_solver_type type)
{
    linear_solver_helper(m, pivot, lda, x, type);
}

//-----------------------------------------------------------------------------
void linear_solver(double* m, quarisma_int* pivot, quarisma_int lda, double* x, linear_solver_type type)
{
    linear_solver_helper(m, pivot, lda, x, type);
}
}  // namespace linalg
