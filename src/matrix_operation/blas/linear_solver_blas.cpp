#include "include/matrix_operation/linear_solver.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include <include/logging.h>

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
// LAPACKE-dependent — see CMakeLists.txt and
// cholesky_decomposition.cpp.
#include <lapacke.h>

#include "include/common/workspace_arena.h"
#endif

// BLAS/LAPACKE specializations for linear_solver.h (the native backend lives in
// linear_solver.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
// on; the #if below is kept as a second guard.
namespace linalg
{
#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

// Same checks as linear_solver.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_solver_inputs(const void* m, const void* x, linalg_int lda)
{
    LOGGING_CHECK(m != nullptr && x != nullptr, "solver_backend::apply: m and x must not be null");
    LOGGING_CHECK(lda > 0, "solver_backend::apply: lda must be positive", lda);
}

namespace
{
// See lu_decomposition.cpp: a generic LAPACKE's lapack_int does not
// necessarily match linalg_int, so the pivot buffer (already populated
// with linalg_int-widened values by lu_decomposition's own blas backend)
// is narrowed back down for this call. `ipiv` (n lapack_int) is
// caller-owned scratch, carved from the arena below.
void to_lapack_pivots(const linalg_int* pivot, linalg_int n, lapack_int* ipiv)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        ipiv[i] = static_cast<lapack_int>(pivot[i]);
    }
}

// Layout: lu_decomposition's own nested workspace (0 unless LU_LINEAR_SOLVER
// needs to factor), then the `ipiv` narrowing buffer (needed whenever getrs
// is reached, i.e. both LU_LINEAR_SOLVER and LU_UPFRONT_LINEAR_SOLVER).
template <typename T> size_t solver_blas_workspace_size(linalg_int lda, linear_solver_type type)
{
    detail::workspace_arena arena;
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        arena.next<unsigned char>(lu_backend<T, kActiveBackend>::workspace_size(lda));
        arena.next<lapack_int>(static_cast<size_t>(lda));
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        arena.next<lapack_int>(static_cast<size_t>(lda));
        break;
    default:
        break;
    }
    return arena.total_size();
}

}  // namespace

template <>
void solver_backend<float, linalg_backend::blas_lapacke>::apply(float* m,
    linalg_int*                                                        pivot,
    linalg_int                                                         lda,
    float*                                                             x,
    linear_solver_type                                                 type,
    void*                                                              workspace,
    size_t                                                             workspace_size)
{
    check_solver_inputs(m, x, lda);
    LOGGING_CHECK(workspace_size >= solver_blas_workspace_size<float>(lda, type),
        "linear_solver: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws =
            arena.next<unsigned char>(lu_backend<float, kActiveBackend>::workspace_size(lda));
        auto* ipiv = arena.next<lapack_int>(static_cast<size_t>(lda));
        if (lu_backend<float, kActiveBackend>::apply(
                m, lda, pivot, lu_ws, lu_backend<float, kActiveBackend>::workspace_size(lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv, x, 1);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_backend<float, kActiveBackend>::apply(
                m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', n, 1, m, n, x, 1);
        }
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        auto* ipiv = arena.next<lapack_int>(static_cast<size_t>(lda));
        to_lapack_pivots(pivot, lda, ipiv);
        LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv, x, 1);
        break;
    }
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', n, 1, m, n, x, 1);
        break;
    }
}

template <>
void solver_backend<double, linalg_backend::blas_lapacke>::apply(double* m,
    linalg_int*                                                          pivot,
    linalg_int                                                           lda,
    double*                                                              x,
    linear_solver_type                                                   type,
    void*                                                                workspace,
    size_t                                                               workspace_size)
{
    check_solver_inputs(m, x, lda);
    LOGGING_CHECK(workspace_size >= solver_blas_workspace_size<double>(lda, type),
        "linear_solver: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws =
            arena.next<unsigned char>(lu_backend<double, kActiveBackend>::workspace_size(lda));
        auto* ipiv = arena.next<lapack_int>(static_cast<size_t>(lda));
        if (lu_backend<double, kActiveBackend>::apply(
                m, lda, pivot, lu_ws, lu_backend<double, kActiveBackend>::workspace_size(lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv, x, 1);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_backend<double, kActiveBackend>::apply(
                m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, 1, m, n, x, 1);
        }
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        auto* ipiv = arena.next<lapack_int>(static_cast<size_t>(lda));
        to_lapack_pivots(pivot, lda, ipiv);
        LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv, x, 1);
        break;
    }
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, 1, m, n, x, 1);
        break;
    }
}

template <>
size_t solver_backend<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return solver_blas_workspace_size<float>(lda, type);
}

template <>
size_t solver_backend<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return solver_blas_workspace_size<double>(lda, type);
}

#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
}  // namespace linalg
