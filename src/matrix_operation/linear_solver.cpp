#include "include/matrix_operation/linear_solver.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include <include/logging.h>

#include <algorithm>

#include "include/common/macros.h"

namespace linalg
{

// Null/positivity checks shared by every linear_solver<T, Backend>::apply
// below, so callers (tests included) get the same validation regardless of
// which backend is active. There is no non-detail linear_solver() wrapper
// to put this in any more — callers use
// linear_solver<T, kActiveBackend>::apply directly.
static void check_solver_inputs(const void* m, const void* x, linalg_int lda)
{
    LOGGING_CHECK(m != nullptr && x != nullptr, "linear_solver::apply: m and x must not be null");
    LOGGING_CHECK(lda > 0, "linear_solver::apply: lda must be positive", lda);
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/linear_solver_mkl.cpp and
// blas/linear_solver_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

#define A(i, j) m[(i) * lda + (j)]

template <typename T>
void lu_solve(T* m, LINALG_UNUSED const linalg_int* pivot, linalg_int lda, T* x)
{
#if LINALG_HAS_LU_PIVOTING
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
        // Scalar lu_decomposition needs no scratch of its own.
        if (lu_decomposition<T, kActiveBackend>::apply(m, lda, pivot, nullptr, 0))
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
        if (cholesky_decomposition<T, kActiveBackend>::apply(
                m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
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

template <>
void linear_solver<float, linalg_backend::native>::apply(float* m,
    linalg_int*                                                  pivot,
    linalg_int                                                   lda,
    float*                                                       x,
    linear_solver_type                                           type,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    check_solver_inputs(m, x, lda);
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

template <>
void linear_solver<double, linalg_backend::native>::apply(double* m,
    linalg_int*                                                    pivot,
    linalg_int                                                     lda,
    double*                                                        x,
    linear_solver_type                                             type,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    check_solver_inputs(m, x, lda);
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

template <>
size_t linear_solver<float, linalg_backend::native>::workspace_size(
    linalg_int /*lda*/, linear_solver_type /*type*/)
{
    return 0;
}

template <>
size_t linear_solver<double, linalg_backend::native>::workspace_size(
    linalg_int /*lda*/, linear_solver_type /*type*/)
{
    return 0;
}
}  // namespace linalg
