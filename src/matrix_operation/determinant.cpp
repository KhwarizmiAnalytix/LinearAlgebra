#include "include/matrix_operation/determinant.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/common/linalg_checks.h"

#include "include/common/linalg_profiler.h"

namespace linalg
{

// determinant just reads the diagonal of an already-decomposed
// buffer — there is no vendor-library "determinant" routine to call into,
// so this always runs directly against whatever lu_decomposition /
// cholesky_decomposition already wrote into `m`. The diagonal entry (i, i)
// is at i * (lda + 1); walking it via a running pointer stepped by that
// stride avoids recomputing the multiply on every iteration.
template <typename T> T lu_determinant(T* m, linalg_int lda)
{
    T        det  = m[0];
    const T* diag = m + lda + 1;
    for (linalg_int i = 1; i < lda; ++i, diag += lda + 1)
    {
        det *= *diag;
    }
    return det;
}

template <typename T> T cholesky_determinant(T* m, linalg_int lda)
{
    T        det  = m[0];
    const T* diag = m + lda + 1;
    for (linalg_int i = 1; i < lda; ++i, diag += lda + 1)
    {
        det *= *diag;
    }
    return det;
}

template <typename T, linalg_backend Backend>
size_t matrix_determinant_workspace_size_helper(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        return lu_decomposition<T, Backend>::workspace_size(lda);
    default:
        return 0;
    }
}

// m and pivot are __restrict: they're always two distinct caller-owned
// buffers (determinant.h documents pivot as the LU pivot array, separate
// from the matrix m).
template <typename T, linalg_backend Backend>
T matrix_determinant_helper(T* __restrict m,
    linalg_int* __restrict     pivot,
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
        LINALG_CHECK(workspace_size >= (matrix_determinant_workspace_size_helper<T, Backend>(lda, type)),
            "determinant: workspace_size too small");
        if (lu_decomposition<T, Backend>::apply(m, lda, pivot, workspace, workspace_size))
        {
            ret = lu_determinant(m, lda);
        }
    }
        return ret;

    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant(m, lda);

    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    {
        if (cholesky_decomposition<T, Backend>::apply(
                m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            ret = cholesky_determinant(m, lda);
        }
    }
        return ret;

    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant(m, lda);
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
size_t determinant<float, linalg_backend::native>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<float, linalg_backend::native>(lda, type);
}

template <>
size_t determinant<double, linalg_backend::native>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<double, linalg_backend::native>(lda, type);
}

template <>
float determinant<float, linalg_backend::native>::apply(float* __restrict m,
    linalg_int* __restrict                                                    pivot,
    linalg_int                                                                lda,
    linear_solver_type                                                        type,
    void*                                                                     workspace,
    size_t                                                                    workspace_size)
{
    LINALG_SCOPE("determinant/native<float>");
    LINALG_CHECK(m != nullptr, "determinant::apply: m must not be null");
    LINALG_CHECK(lda > 0, "determinant::apply: lda must be positive", lda);
    return matrix_determinant_helper<float, linalg_backend::native>(m, pivot, lda, type, workspace, workspace_size);
}

template <>
double determinant<double, linalg_backend::native>::apply(double* __restrict m,
    linalg_int* __restrict                                                       pivot,
    linalg_int                                                                   lda,
    linear_solver_type                                                           type,
    void*                                                                        workspace,
    size_t                                                                       workspace_size)
{
    LINALG_SCOPE("determinant/native<double>");
    LINALG_CHECK(m != nullptr, "determinant::apply: m must not be null");
    LINALG_CHECK(lda > 0, "determinant::apply: lda must be positive", lda);
    return matrix_determinant_helper<double, linalg_backend::native>(m, pivot, lda, type, workspace, workspace_size);
}

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

template <>
size_t determinant<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<float, linalg_backend::blas_lapacke>(lda, type);
}

template <>
size_t determinant<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<double, linalg_backend::blas_lapacke>(lda, type);
}

template <>
float determinant<float, linalg_backend::blas_lapacke>::apply(float* __restrict m,
    linalg_int* __restrict                                                           pivot,
    linalg_int                                                                       lda,
    linear_solver_type                                                               type,
    void*                                                                            workspace,
    size_t                                                                           workspace_size)
{
    LINALG_SCOPE("determinant/blas_lapacke<float>");
    LINALG_CHECK(m != nullptr, "determinant::apply: m must not be null");
    LINALG_CHECK(lda > 0, "determinant::apply: lda must be positive", lda);
    return matrix_determinant_helper<float, linalg_backend::blas_lapacke>(m, pivot, lda, type, workspace, workspace_size);
}

template <>
double determinant<double, linalg_backend::blas_lapacke>::apply(double* __restrict m,
    linalg_int* __restrict                                                              pivot,
    linalg_int                                                                          lda,
    linear_solver_type                                                                  type,
    void*                                                                               workspace,
    size_t                                                                              workspace_size)
{
    LINALG_SCOPE("determinant/blas_lapacke<double>");
    LINALG_CHECK(m != nullptr, "determinant::apply: m must not be null");
    LINALG_CHECK(lda > 0, "determinant::apply: lda must be positive", lda);
    return matrix_determinant_helper<double, linalg_backend::blas_lapacke>(m, pivot, lda, type, workspace, workspace_size);
}

#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#if LINALG_HAS_MKL

template <>
size_t determinant<float, linalg_backend::mkl>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<float, linalg_backend::mkl>(lda, type);
}

template <>
size_t determinant<double, linalg_backend::mkl>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return matrix_determinant_workspace_size_helper<double, linalg_backend::mkl>(lda, type);
}

template <>
float determinant<float, linalg_backend::mkl>::apply(float* __restrict m,
    linalg_int* __restrict                                                  pivot,
    linalg_int                                                              lda,
    linear_solver_type                                                      type,
    void*                                                                   workspace,
    size_t                                                                  workspace_size)
{
    LINALG_SCOPE("determinant/mkl<float>");
    LINALG_CHECK(m != nullptr, "determinant::apply: m must not be null");
    LINALG_CHECK(lda > 0, "determinant::apply: lda must be positive", lda);
    return matrix_determinant_helper<float, linalg_backend::mkl>(m, pivot, lda, type, workspace, workspace_size);
}

template <>
double determinant<double, linalg_backend::mkl>::apply(double* __restrict m,
    linalg_int* __restrict                                                     pivot,
    linalg_int                                                                 lda,
    linear_solver_type                                                         type,
    void*                                                                      workspace,
    size_t                                                                     workspace_size)
{
    LINALG_SCOPE("determinant/mkl<double>");
    LINALG_CHECK(m != nullptr, "determinant::apply: m must not be null");
    LINALG_CHECK(lda > 0, "determinant::apply: lda must be positive", lda);
    return matrix_determinant_helper<double, linalg_backend::mkl>(m, pivot, lda, type, workspace, workspace_size);
}

#endif  // LINALG_HAS_MKL

}  // namespace linalg
