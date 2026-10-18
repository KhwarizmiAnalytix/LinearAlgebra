#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#include "include/matrix_operation/linear_solver.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/common/linalg_checks.h"

// LAPACKE-dependent — see CMakeLists.txt and
// cholesky_decomposition.cpp.
#include <complex>   // needed by OpenBLAS lapack.h on MSVC (no _Complex support)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4190) // C-linkage returning std::complex<> (OpenBLAS lapacke.h)
#endif
#include <lapacke.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "include/common/workspace_arena.h"

#include "include/common/linalg_profiler.h"

// BLAS/LAPACKE specializations for linear_solver.h (the native backend lives in
// linear_solver.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
// on; the #if below is kept as a second guard.
namespace linalg
{
// Same checks as linear_solver.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_solver_inputs(const void* m, const void* x, linalg_int lda)
{
    LINALG_CHECK(m != nullptr && x != nullptr, "linear_solver::apply: m and x must not be null");
    LINALG_CHECK(lda > 0, "linear_solver::apply: lda must be positive", lda);
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
        arena.next<unsigned char>(lu_decomposition<T, linalg_backend::blas_lapacke>::workspace_size(lda));
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
void linear_solver<float, linalg_backend::blas_lapacke>::apply(float* m,
    linalg_int*                                                        pivot,
    linalg_int                                                         lda,
    float*                                                             x,
    linear_solver_type                                                 type,
    void*                                                              workspace,
    size_t                                                             workspace_size)
{
    LINALG_SCOPE("linear_solver/blas_lapacke<float>");
    check_solver_inputs(m, x, lda);
    LINALG_CHECK(workspace_size >= solver_blas_workspace_size<float>(lda, type),
        "linear_solver: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws =
            arena.next<unsigned char>(lu_decomposition<float, linalg_backend::blas_lapacke>::workspace_size(lda));
        auto* ipiv = arena.next<lapack_int>(static_cast<size_t>(lda));
        if (lu_decomposition<float, linalg_backend::blas_lapacke>::apply(
                m, lda, pivot, lu_ws, lu_decomposition<float, linalg_backend::blas_lapacke>::workspace_size(lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv, x, 1);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition<float, linalg_backend::blas_lapacke>::apply(
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
void linear_solver<double, linalg_backend::blas_lapacke>::apply(double* m,
    linalg_int*                                                          pivot,
    linalg_int                                                           lda,
    double*                                                              x,
    linear_solver_type                                                   type,
    void*                                                                workspace,
    size_t                                                               workspace_size)
{
    LINALG_SCOPE("linear_solver/blas_lapacke<double>");
    check_solver_inputs(m, x, lda);
    LINALG_CHECK(workspace_size >= solver_blas_workspace_size<double>(lda, type),
        "linear_solver: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws =
            arena.next<unsigned char>(lu_decomposition<double, linalg_backend::blas_lapacke>::workspace_size(lda));
        auto* ipiv = arena.next<lapack_int>(static_cast<size_t>(lda));
        if (lu_decomposition<double, linalg_backend::blas_lapacke>::apply(
                m, lda, pivot, lu_ws, lu_decomposition<double, linalg_backend::blas_lapacke>::workspace_size(lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv, x, 1);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition<double, linalg_backend::blas_lapacke>::apply(
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
size_t linear_solver<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return solver_blas_workspace_size<float>(lda, type);
}

template <>
size_t linear_solver<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return solver_blas_workspace_size<double>(lda, type);
}

}  // namespace linalg
#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
