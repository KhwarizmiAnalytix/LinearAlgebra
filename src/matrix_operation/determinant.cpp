#include "include/matrix_operation/determinant.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/logging.h"

namespace linalg
{

// determinant just reads the diagonal of an already-decomposed
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
        return lu_decomposition<T, linalg_backend::native>::workspace_size(lda);
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
            "determinant: workspace_size too small");
        if (lu_decomposition<T, linalg_backend::native>::apply(m, lda, pivot, workspace, workspace_size))
        {
            ret = lu_determinant(m, lda);
        }
    }
        return ret;

    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant(m, lda);

    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    {
        if (cholesky_decomposition<T, linalg_backend::native>::apply(
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

// determinant: always uses the scalar path (no vendor-library
// determinant routine), so only `native` is specialized — callers always name
// it explicitly.
template <>
size_t determinant<float, linalg_backend::native>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<float>(lda, type);
}

template <>
size_t determinant<double, linalg_backend::native>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<double>(lda, type);
}

template <>
float determinant<float, linalg_backend::native>::apply(float* m,
    linalg_int*                                                               pivot,
    linalg_int                                                                lda,
    linear_solver_type                                                        type,
    void*                                                                     workspace,
    size_t                                                                    workspace_size)
{
    LOGGING_CHECK(m != nullptr, "determinant::apply: m must not be null");
    LOGGING_CHECK(lda > 0, "determinant::apply: lda must be positive", lda);
    return matrix_determinant_helper(m, pivot, lda, type, workspace, workspace_size);
}

template <>
double determinant<double, linalg_backend::native>::apply(double* m,
    linalg_int*                                                                  pivot,
    linalg_int                                                                   lda,
    linear_solver_type                                                           type,
    void*                                                                        workspace,
    size_t                                                                       workspace_size)
{
    LOGGING_CHECK(m != nullptr, "determinant::apply: m must not be null");
    LOGGING_CHECK(lda > 0, "determinant::apply: lda must be positive", lda);
    return matrix_determinant_helper(m, pivot, lda, type, workspace, workspace_size);
}

}  // namespace linalg
