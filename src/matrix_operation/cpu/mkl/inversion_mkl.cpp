#if LINALG_HAS_MKL

#include "include/matrix_operation/inversion.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/common/linalg_checks.h"

#include <mkl.h>

#include "include/common/workspace_arena.h"

#include "include/common/linalg_profiler.h"

// MKL specializations for inversion.h (the native backend lives in
// inversion.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
// on; the #if below is kept as a second guard.
namespace linalg
{

// Same checks as inversion.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_invert_inputs(const void* m, linalg_int lda)
{
    LINALG_CHECK(m != nullptr, "inversion::apply: m must not be null");
    LINALG_CHECK(lda > 0, "inversion::apply: lda must be positive", lda);
}

namespace
{
// getri needs its own LAPACK work array (unlike getrf/potrf/potri, which
// need none); lwork is queried with a dummy lwork=-1 call per LAPACK
// convention (the optimal size comes back in work[0]).
template <typename T, typename GetriWorkFn>
lapack_int getri_lwork(GetriWorkFn getri_work, lapack_int n)
{
    T query[1];
    getri_work(LAPACK_ROW_MAJOR, n, nullptr, n, nullptr, query, -1);
    return static_cast<lapack_int>(query[0]);
}

// Layout: lu_decomposition's own nested workspace (0 unless LU_LINEAR_SOLVER
// needs to factor), then getri's work array (needed for both LU types).
template <typename T, typename GetriWorkFn>
size_t invert_mkl_workspace_size_impl(
    GetriWorkFn getri_work, linalg_int lda, linear_solver_type type)
{
    detail::workspace_arena arena;
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        arena.next<unsigned char>(lu_decomposition<T, linalg_backend::mkl>::workspace_size(lda));
        arena.next<T>(
            static_cast<size_t>(getri_lwork<T>(getri_work, static_cast<lapack_int>(lda))));
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        arena.next<T>(
            static_cast<size_t>(getri_lwork<T>(getri_work, static_cast<lapack_int>(lda))));
        break;
    default:
        break;
    }
    return arena.total_size();
}
}  // namespace

template <>
void inversion<float, linalg_backend::mkl>::apply(float* m,
    linalg_int*                                               pivot,
    linalg_int                                                lda,
    linear_solver_type                                        type,
    void*                                                     workspace,
    size_t                                                    workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("inversion/mkl<float>");
    check_invert_inputs(m, lda);
    LINALG_CHECK(
        workspace_size >= invert_mkl_workspace_size_impl<float>(LAPACKE_sgetri_work, lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws =
            arena.next<unsigned char>(lu_decomposition<float, linalg_backend::mkl>::workspace_size(lda));
        const auto lwork = getri_lwork<float>(LAPACKE_sgetri_work, n);
        auto*      work  = arena.next<float>(static_cast<size_t>(lwork));
        if (lu_decomposition<float, linalg_backend::mkl>::apply(
                m, lda, pivot, lu_ws, lu_decomposition<float, linalg_backend::mkl>::workspace_size(lda)))
        {
            LAPACKE_sgetri_work(LAPACK_ROW_MAJOR, n, m, n, pivot, work, lwork);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition<float, linalg_backend::mkl>::apply(
                m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
        }
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        const auto lwork = getri_lwork<float>(LAPACKE_sgetri_work, n);
        auto*      work  = arena.next<float>(static_cast<size_t>(lwork));
        LAPACKE_sgetri_work(LAPACK_ROW_MAJOR, n, m, n, pivot, work, lwork);
        break;
    }
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
        break;
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
void inversion<double, linalg_backend::mkl>::apply(double* m,
    linalg_int*                                                 pivot,
    linalg_int                                                  lda,
    linear_solver_type                                          type,
    void*                                                       workspace,
    size_t                                                      workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("inversion/mkl<double>");
    check_invert_inputs(m, lda);
    LINALG_CHECK(
        workspace_size >= invert_mkl_workspace_size_impl<double>(LAPACKE_dgetri_work, lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws =
            arena.next<unsigned char>(lu_decomposition<double, linalg_backend::mkl>::workspace_size(lda));
        const auto lwork = getri_lwork<double>(LAPACKE_dgetri_work, n);
        auto*      work  = arena.next<double>(static_cast<size_t>(lwork));
        if (lu_decomposition<double, linalg_backend::mkl>::apply(
                m, lda, pivot, lu_ws, lu_decomposition<double, linalg_backend::mkl>::workspace_size(lda)))
        {
            LAPACKE_dgetri_work(LAPACK_ROW_MAJOR, n, m, n, pivot, work, lwork);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition<double, linalg_backend::mkl>::apply(
                m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
        }
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        const auto lwork = getri_lwork<double>(LAPACKE_dgetri_work, n);
        auto*      work  = arena.next<double>(static_cast<size_t>(lwork));
        LAPACKE_dgetri_work(LAPACK_ROW_MAJOR, n, m, n, pivot, work, lwork);
        break;
    }
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
        break;
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
size_t inversion<float, linalg_backend::mkl>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return invert_mkl_workspace_size_impl<float>(LAPACKE_sgetri_work, lda, type);
}

template <>
size_t inversion<double, linalg_backend::mkl>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return invert_mkl_workspace_size_impl<double>(LAPACKE_dgetri_work, lda, type);
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
