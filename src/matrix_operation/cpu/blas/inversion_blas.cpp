#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#include "include/matrix_operation/inversion.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/common/linalg_checks.h"

#include <complex>   // needed by OpenBLAS lapack.h on MSVC (no _Complex support)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4190) // C-linkage returning std::complex<> (OpenBLAS lapacke.h)
#endif
#include <lapacke.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "include/common/workspace_arena.h"

#include "include/common/linalg_profiler.h"

// BLAS/LAPACKE specializations for inversion.h (the native backend lives in
// inversion.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
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
// See lu_decomposition.cpp's blas_lapack branch. `ipiv` (n lapack_int) is
// caller-owned scratch, carved from the arena below.
void to_lapack_pivots(const linalg_int* pivot, linalg_int n, lapack_int* ipiv)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        ipiv[i] = static_cast<lapack_int>(pivot[i]);
    }
}

template <typename T, typename GetriWorkFn>
lapack_int getri_lwork(GetriWorkFn getri_work, lapack_int n)
{
    T query[1];
    getri_work(LAPACK_ROW_MAJOR, n, nullptr, n, nullptr, query, -1);
    return static_cast<lapack_int>(query[0]);
}

// Layout: lu_decomposition's own nested workspace (0 unless LU_LINEAR_SOLVER
// needs to factor), the `ipiv` narrowing buffer, then getri's work array
// (both needed for either LU type).
template <typename T, typename GetriWorkFn>
size_t invert_blas_workspace_size_impl(
    GetriWorkFn getri_work, linalg_int lda, linear_solver_type type)
{
    detail::workspace_arena arena;
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        arena.next<unsigned char>(lu_decomposition<T, linalg_backend::blas_lapacke>::workspace_size(lda));
        arena.next<lapack_int>(static_cast<size_t>(lda));
        arena.next<T>(
            static_cast<size_t>(getri_lwork<T>(getri_work, static_cast<lapack_int>(lda))));
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        arena.next<lapack_int>(static_cast<size_t>(lda));
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
void inversion<float, linalg_backend::blas_lapacke>::apply(float* m,
    linalg_int*                                                        pivot,
    linalg_int                                                         lda,
    linear_solver_type                                                 type,
    void*                                                              workspace,
    size_t                                                             workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("inversion/blas_lapacke<float>");
    check_invert_inputs(m, lda);
    LINALG_CHECK(
        workspace_size >= invert_blas_workspace_size_impl<float>(LAPACKE_sgetri_work, lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws =
            arena.next<unsigned char>(lu_decomposition<float, linalg_backend::blas_lapacke>::workspace_size(lda));
        auto*      ipiv  = arena.next<lapack_int>(static_cast<size_t>(lda));
        const auto lwork = getri_lwork<float>(LAPACKE_sgetri_work, n);
        auto*      work  = arena.next<float>(static_cast<size_t>(lwork));
        if (lu_decomposition<float, linalg_backend::blas_lapacke>::apply(
                m, lda, pivot, lu_ws, lu_decomposition<float, linalg_backend::blas_lapacke>::workspace_size(lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_sgetri_work(LAPACK_ROW_MAJOR, n, m, n, ipiv, work, lwork);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition<float, linalg_backend::blas_lapacke>::apply(
                m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
        }
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        auto*      ipiv  = arena.next<lapack_int>(static_cast<size_t>(lda));
        const auto lwork = getri_lwork<float>(LAPACKE_sgetri_work, n);
        auto*      work  = arena.next<float>(static_cast<size_t>(lwork));
        to_lapack_pivots(pivot, lda, ipiv);
        LAPACKE_sgetri_work(LAPACK_ROW_MAJOR, n, m, n, ipiv, work, lwork);
        break;
    }
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
        break;
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
void inversion<double, linalg_backend::blas_lapacke>::apply(double* m,
    linalg_int*                                                          pivot,
    linalg_int                                                           lda,
    linear_solver_type                                                   type,
    void*                                                                workspace,
    size_t                                                               workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("inversion/blas_lapacke<double>");
    check_invert_inputs(m, lda);
    LINALG_CHECK(
        workspace_size >= invert_blas_workspace_size_impl<double>(LAPACKE_dgetri_work, lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws =
            arena.next<unsigned char>(lu_decomposition<double, linalg_backend::blas_lapacke>::workspace_size(lda));
        auto*      ipiv  = arena.next<lapack_int>(static_cast<size_t>(lda));
        const auto lwork = getri_lwork<double>(LAPACKE_dgetri_work, n);
        auto*      work  = arena.next<double>(static_cast<size_t>(lwork));
        if (lu_decomposition<double, linalg_backend::blas_lapacke>::apply(
                m, lda, pivot, lu_ws, lu_decomposition<double, linalg_backend::blas_lapacke>::workspace_size(lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_dgetri_work(LAPACK_ROW_MAJOR, n, m, n, ipiv, work, lwork);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition<double, linalg_backend::blas_lapacke>::apply(
                m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
        }
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
    {
        auto*      ipiv  = arena.next<lapack_int>(static_cast<size_t>(lda));
        const auto lwork = getri_lwork<double>(LAPACKE_dgetri_work, n);
        auto*      work  = arena.next<double>(static_cast<size_t>(lwork));
        to_lapack_pivots(pivot, lda, ipiv);
        LAPACKE_dgetri_work(LAPACK_ROW_MAJOR, n, m, n, ipiv, work, lwork);
        break;
    }
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
        break;
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
size_t inversion<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return invert_blas_workspace_size_impl<float>(LAPACKE_sgetri_work, lda, type);
}

template <>
size_t inversion<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return invert_blas_workspace_size_impl<double>(LAPACKE_dgetri_work, lda, type);
}
}  // namespace linalg

#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
