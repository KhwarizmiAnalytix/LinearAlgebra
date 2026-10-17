#include "include/matrix_operation/linear_solver.h"

#include "include/common/linalg_checks.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"

#include <algorithm>

#include "include/common/linalg_macros.h"

#include "include/common/linalg_profiler.h"

namespace linalg
{

// Null/positivity checks shared by every linear_solver<T, Backend>::apply
// below, so callers (tests included) get the same validation regardless of
// which backend is active. There is no non-detail linear_solver() wrapper
// to put this in any more — callers use
// linear_solver<T, linalg_backend::native>::apply directly.
static void check_solver_inputs(const void* m, const void* x, linalg_int lda)
{
    LINALG_CHECK(m != nullptr && x != nullptr, "linear_solver::apply: m and x must not be null");
    LINALG_CHECK(lda > 0, "linear_solver::apply: lda must be positive", lda);
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/linear_solver_mkl.cpp and
// blas/linear_solver_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

// m, pivot, and x are __restrict: linear_solver.h documents them as three
// distinct caller-owned buffers (the matrix/factors, the pivot array, and
// the in/out right-hand side), never aliasing each other.
template <typename T>
void lu_solve(T* __restrict m, LINALG_UNUSED const linalg_int* __restrict pivot, linalg_int lda,
    T* __restrict x)
{
#if LINALG_HAS_LU_PIVOTING
    for (int i = 0; i < lda; i++)
    {
        std::swap(x[i], x[pivot[i] - 1]);
    }
#endif
    // row_i is hoisted once per i and indexed by j below, instead of m's
    // row base (i * lda) being recomputed for every j.
    for (linalg_int i = 0; i < lda; i++)
    {
        const T* const row_i = m + i * lda;
        auto            sum  = x[i];  //
        for (linalg_int j = 0; j < i; ++j)
        {
            sum -= row_i[j] * x[j];
        }

        x[i] = sum;
    }

    for (linalg_int i = lda - 1; i >= 0; i--)
    {
        const T* const row_i = m + i * lda;
        auto            sum  = x[i];  //
        for (linalg_int j = i + 1; j < lda; ++j)
        {
            sum -= row_i[j] * x[j];
        }

        x[i] = sum / row_i[i];
    }
}

// m and x are __restrict, same reasoning as lu_solve above.
template <typename T> void cholesky_solve(T* __restrict m, linalg_int lda, T* __restrict x)
{
    // Forward substitution reads row i (contiguous): row_i is hoisted once
    // per i, reused for every j.
    for (linalg_int i = 0; i < lda; i++)
    {
        const T* const row_i = m + i * lda;
        auto            sum  = x[i];
        for (linalg_int j = 0; j < i; ++j)
        {
            sum -= row_i[j] * x[j];
        }

        x[i] = sum / row_i[i];
    }

    // Back substitution reads column i (stride lda, since this walks the
    // transposed triangle): col_ptr steps by lda per j instead of
    // recomputing j * lda + i from scratch each time.
    for (linalg_int i = lda - 1; i >= 0; i--)
    {
        auto     sum     = x[i];  //
        const T* col_ptr = m + (i + 1) * lda + i;
        for (linalg_int j = i + 1; j < lda; ++j, col_ptr += lda)
        {
            sum -= (*col_ptr) * x[j];
        }

        x[i] = sum / m[i * lda + i];
    }
}

// m, pivot, and x are __restrict, same reasoning as lu_solve above.
template <typename T>
void linear_solver_scalar_impl(T* __restrict m, linalg_int* __restrict pivot, linalg_int lda,
    T* __restrict x, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        // Scalar lu_decomposition needs no scratch of its own.
        if (lu_decomposition<T, linalg_backend::native>::apply(m, lda, pivot, nullptr, 0))
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
        if (cholesky_decomposition<T, linalg_backend::native>::apply(
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
void linear_solver<float, linalg_backend::native>::apply(float* __restrict m,
    linalg_int* __restrict                                      pivot,
    linalg_int                                                  lda,
    float* __restrict                                           x,
    linear_solver_type                                          type,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    LINALG_SCOPE("linear_solver/native<float>");
    check_solver_inputs(m, x, lda);
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

template <>
void linear_solver<double, linalg_backend::native>::apply(double* __restrict m,
    linalg_int* __restrict                                        pivot,
    linalg_int                                                    lda,
    double* __restrict                                            x,
    linear_solver_type                                            type,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    LINALG_SCOPE("linear_solver/native<double>");
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
