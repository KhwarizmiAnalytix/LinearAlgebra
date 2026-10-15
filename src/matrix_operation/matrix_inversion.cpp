#include "include/matrix_operation/matrix_inversion.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include <include/logging.h>

#include <algorithm>

#include "include/common/macros.h"
#include "include/common/workspace_arena.h"

namespace linalg
{

// Null checks shared by every invert_backend<T, Backend>::apply below, so
// callers (tests included) get the same validation regardless of which
// backend is active. There is no non-detail matrix_invert() wrapper to put
// this in any more — callers use invert_backend<T, kActiveBackend>::apply
// directly.
static void check_invert_inputs(const void* m, linalg_int lda)
{
    LOGGING_CHECK(m != nullptr, "invert_backend::apply: m must not be null");
    LOGGING_CHECK(lda > 0, "invert_backend::apply: lda must be positive", lda);
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/matrix_inversion_mkl.cpp and
// blas/matrix_inversion_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

template <typename T> size_t invert_scalar_workspace_size(linalg_int lda, linear_solver_type type)
{
    detail::workspace_arena arena;
    const auto              n = static_cast<size_t>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        // lu_invert keeps a double-precision inverse and column scratch for
        // both float and double, matching the previous std::vector<double>.
        arena.next<double>(n * n);
        arena.next<double>(n);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        arena.next<T>(n);
        break;
    default:
        break;
    }
    return arena.total_size();
}

namespace
{

#define A(i, j) m[(i) * lda + (j)]
#define IA(i, j) inv_m[(i) * lda + (j)]

template <typename T>
void lu_invert(
    T* m, LINALG_UNUSED const linalg_int* pivot, linalg_int lda, double* inv_m, double* work)
{
    using size_type = linalg_int;
    using value_t   = T;
    for (size_type j = 0; j < lda; ++j)
    {
        for (size_type i = 0; i < lda; ++i)
        {
            value_t sum = i == j ? 1.0 : 0.0;

            for (size_type k = 0; k < i; ++k)
            {
                sum -= A(i, k) * work[k];
            }

            work[i] = static_cast<value_t>(sum);
        }

        for (size_type i = lda - 1; i >= 0; --i)
        {
            auto& sum = work[i];

            for (size_type k = i + 1; k < lda; ++k)
            {
                sum -= A(i, k) * work[k];
            }

            sum /= A(i, i);
        }

        for (size_type i = 0; i < lda; ++i)
        {
            IA(i, j) = work[i];
        }
    }
    for (size_type j = 0; j < lda; ++j)
    {
        for (size_type i = 0; i < lda; ++i)
        {
            A(i, j) = IA(i, j);
        }
    }
#if LINALG_HAS_LU_PIVOTING
    for (size_type i = lda - 1; i >= 0; --i)
    {
        size_type p = pivot[i] - 1;
        if (p != i)
        {
            for (size_type j = 0; j < lda; ++j)
            {
                auto& a_ij = A(j, i);
                auto& a_pj = A(j, p);
                std::swap(a_ij, a_pj);
            }
        }
    }
#endif
}

template <typename T> void cholesky_invert(T* m, linalg_int lda, T* work)
{
    using value_t   = T;
    using size_type = linalg_int;

    for (size_type j = 0; j < lda; ++j)
    {
        // Compute L^-1
        for (size_type i = j; i < lda; ++i)
        {
            auto sum = static_cast<value_t>((i == j) ? 1.0 : 0.0);
            for (size_type k = j; k < i; ++k)
            {
                sum -= A(i, k) * A(k, j);
            }
            A(i, j) = sum / A(i, i);
        }

        // Compute (L^T)^-1 * L^-1 for the j-th column
        for (size_type i = lda - 1; i >= 0; --i)
        {
            auto sum = static_cast<value_t>(0.0);
            for (size_type k = i; k < lda; ++k)
            {
                sum += A(k, i) * A(k, j);
            }
            work[i] = sum;
        }

        // Copy the result back to the original matrix
        for (size_type i = 0; i <= j; ++i)
        {
            A(i, j) = work[i];
        }
    }

    // Fill in the lower triangular part
    for (size_type i = 1; i < lda; ++i)
    {
        for (size_type j = 0; j < i; ++j)
        {
            A(i, j) = A(j, i);
        }
    }
}

#undef A
#undef IA

template <typename T>
void matrix_invert_scalar_impl(T* m,
    linalg_int*                   pivot,
    linalg_int                    lda,
    linear_solver_type            type,
    void*                         workspace,
    size_t                        workspace_size)
{
    LOGGING_CHECK(workspace_size >= invert_scalar_workspace_size<T>(lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<size_t>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        auto* inv_m = arena.next<double>(n * n);
        auto* work  = arena.next<double>(n);
        if (type == linear_solver_type::LU_UPFRONT_LINEAR_SOLVER ||
            lu_backend<T, kActiveBackend>::apply(m, lda, pivot, nullptr, 0))
        {
            lu_invert(m, pivot, lda, inv_m, work);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
    {
        auto* work = arena.next<T>(n);
        if (type == linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER ||
            cholesky_backend<T, kActiveBackend>::apply(
                m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            cholesky_invert(m, lda, work);
        }
        break;
    }
    default:
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

}  // namespace

template <>
void invert_backend<float, linalg_backend::native>::apply(float* m,
    linalg_int*                                                  pivot,
    linalg_int                                                   lda,
    linear_solver_type                                           type,
    void*                                                        workspace,
    size_t                                                       workspace_size)
{
    check_invert_inputs(m, lda);
    matrix_invert_scalar_impl(m, pivot, lda, type, workspace, workspace_size);
}

template <>
void invert_backend<double, linalg_backend::native>::apply(double* m,
    linalg_int*                                                    pivot,
    linalg_int                                                     lda,
    linear_solver_type                                             type,
    void*                                                          workspace,
    size_t                                                         workspace_size)
{
    check_invert_inputs(m, lda);
    matrix_invert_scalar_impl(m, pivot, lda, type, workspace, workspace_size);
}

// matrix_determinant just reads the diagonal of an already-decomposed
// buffer — there is no vendor-library "determinant" routine to call into,
// so this always runs directly against whatever lu_decomposition /
// cholesky_decomposition already wrote into `m`.
#define A(i, j) m[(i) * lda + (j)]

template <typename T> T lu_determinant(T* m, linalg_int lda)
{
    T det = m[0];
    for (linalg_int i = 1; i < lda; ++i)
    {
        det *= A(i, i);
    }
    return det;
}

template <typename T> T cholesky_determinant(T* m, linalg_int lda)
{
    T det = m[0];
    for (linalg_int i = 1; i < lda; ++i)
    {
        det *= A(i, i);
    }
    return det;
}

#undef A

template <typename T>
size_t matrix_determinant_workspace_size_helper(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        return lu_backend<T, kActiveBackend>::workspace_size(lda);
    default:
        return 0;
    }
}

template <typename T>
T matrix_determinant_helper(T* m,
    linalg_int*                pivot,
    linalg_int                 lda,
    linear_solver_type         type,
    void*                      workspace,
    size_t                     workspace_size)
{
    T ret = 0.;
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        LOGGING_CHECK(workspace_size >= matrix_determinant_workspace_size_helper<T>(lda, type),
            "matrix_determinant: workspace_size too small");
        if (lu_backend<T, kActiveBackend>::apply(m, lda, pivot, workspace, workspace_size))
        {
            ret = lu_determinant(m, lda);
        }
    }
        return ret;

    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant(m, lda);

    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    {
        if (cholesky_backend<T, kActiveBackend>::apply(
                m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            ret = cholesky_determinant(m, lda);
        }
    }
        return ret;

    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant(m, lda);
    default:
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
size_t invert_backend<float, linalg_backend::native>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return invert_scalar_workspace_size<float>(lda, type);
}

template <>
size_t invert_backend<double, linalg_backend::native>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return invert_scalar_workspace_size<double>(lda, type);
}

// matrix_determinant_backend: always uses the scalar path (no vendor-library
// determinant routine), so only `native` is specialized — callers always name
// it explicitly.
template <>
size_t matrix_determinant_backend<float, linalg_backend::native>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<float>(lda, type);
}

template <>
size_t matrix_determinant_backend<double, linalg_backend::native>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<double>(lda, type);
}

template <>
float matrix_determinant_backend<float, linalg_backend::native>::apply(float* m,
    linalg_int*                                                               pivot,
    linalg_int                                                                lda,
    linear_solver_type                                                        type,
    void*                                                                     workspace,
    size_t                                                                    workspace_size)
{
    LOGGING_CHECK(m != nullptr, "matrix_determinant_backend::apply: m must not be null");
    LOGGING_CHECK(lda > 0, "matrix_determinant_backend::apply: lda must be positive", lda);
    return matrix_determinant_helper(m, pivot, lda, type, workspace, workspace_size);
}

template <>
double matrix_determinant_backend<double, linalg_backend::native>::apply(double* m,
    linalg_int*                                                                  pivot,
    linalg_int                                                                   lda,
    linear_solver_type                                                           type,
    void*                                                                        workspace,
    size_t                                                                       workspace_size)
{
    LOGGING_CHECK(m != nullptr, "matrix_determinant_backend::apply: m must not be null");
    LOGGING_CHECK(lda > 0, "matrix_determinant_backend::apply: lda must be positive", lda);
    return matrix_determinant_helper(m, pivot, lda, type, workspace, workspace_size);
}

}  // namespace linalg
