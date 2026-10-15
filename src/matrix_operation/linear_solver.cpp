#include "include/matrix_operation/linear_solver.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include <include/logging.h>

#if LINALG_HAS_MKL
#include <mkl.h>
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
// LAPACKE-dependent — see CMakeLists.txt and
// cholesky_decomposition.cpp.
#include <lapacke.h>

#include "include/common/workspace_arena.h"
#else
#include <algorithm>

#include "include/common/macros.h"
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// linear_solver() overloads below call — exactly one is ever built.
#if LINALG_HAS_MKL

// No scratch of its own: an ILP64 MKL build's lu_decomposition also needs
// none (see lu_decomposition.cpp), so `workspace`/`workspace_size` here only
// ever forward unchanged into the nested lu_decomposition call.
size_t solver_mkl_workspace_size(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        return lu_decomposition_workspace_size(static_cast<const float*>(nullptr), lda);
    default:
        return 0;
    }
}

void solver_mkl_f32(float* m, linalg_int* pivot, linalg_int lda, float* x,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        if (lu_decomposition(m, lda, pivot, workspace, workspace_size))
        {
            LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
        }
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
        }
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
        break;
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
        break;
    }
}

void solver_mkl_f64(double* m, linalg_int* pivot, linalg_int lda, double* x,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        if (lu_decomposition(m, lda, pivot, workspace, workspace_size))
        {
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
        }
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
        {
            LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
        }
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', lda, 1, m, lda, pivot, x, 1);
        break;
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', lda, 1, m, lda, x, 1);
        break;
    }
}

#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

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
template <typename T>
size_t solver_blas_workspace_size(linalg_int lda, linear_solver_type type)
{
    detail::workspace_arena arena;
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        arena.next<unsigned char>(
            lu_decomposition_workspace_size(static_cast<const T*>(nullptr), lda));
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

void solver_blas_f32(float* m, linalg_int* pivot, linalg_int lda, float* x,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(workspace_size >= solver_blas_workspace_size<float>(lda, type),
        "linear_solver: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws = arena.next<unsigned char>(
            lu_decomposition_workspace_size(static_cast<const float*>(nullptr), lda));
        auto* ipiv = arena.next<lapack_int>(static_cast<size_t>(lda));
        if (lu_decomposition(m, lda, pivot, lu_ws,
                lu_decomposition_workspace_size(static_cast<const float*>(nullptr), lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv, x, 1);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
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

void solver_blas_f64(double* m, linalg_int* pivot, linalg_int lda, double* x,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(workspace_size >= solver_blas_workspace_size<double>(lda, type),
        "linear_solver: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws = arena.next<unsigned char>(
            lu_decomposition_workspace_size(static_cast<const double*>(nullptr), lda));
        auto* ipiv = arena.next<lapack_int>(static_cast<size_t>(lda));
        if (lu_decomposition(m, lda, pivot, lu_ws,
                lu_decomposition_workspace_size(static_cast<const double*>(nullptr), lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, m, n, ipiv, x, 1);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
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

#else

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
        if (lu_decomposition(m, lda, pivot, nullptr, 0))
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
}

}  // namespace

void solver_scalar_f32(float* m, linalg_int* pivot, linalg_int lda, float* x,
    linear_solver_type type, void* /*workspace*/, size_t /*workspace_size*/)
{
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

void solver_scalar_f64(double* m, linalg_int* pivot, linalg_int lda, double* x,
    linear_solver_type type, void* /*workspace*/, size_t /*workspace_size*/)
{
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

#endif

}  // namespace detail

size_t linear_solver_workspace_size(const float* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
#if LINALG_HAS_MKL
    return detail::solver_mkl_workspace_size(lda, type);
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    return detail::solver_blas_workspace_size<float>(lda, type);
#else
    (void)lda;
    (void)type;
    return 0;
#endif
}

size_t linear_solver_workspace_size(const double* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
#if LINALG_HAS_MKL
    return detail::solver_mkl_workspace_size(lda, type);
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    return detail::solver_blas_workspace_size<double>(lda, type);
#else
    (void)lda;
    (void)type;
    return 0;
#endif
}

//-----------------------------------------------------------------------------
void linear_solver(float* m, linalg_int* pivot, linalg_int lda, float* x, linear_solver_type type,
    void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(m != nullptr && x != nullptr, "linear_solver: m and x must not be null");
    LOGGING_CHECK(lda > 0, "linear_solver: lda must be positive", lda);
#if LINALG_HAS_MKL
    detail::solver_mkl_f32(m, pivot, lda, x, type, workspace, workspace_size);
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    detail::solver_blas_f32(m, pivot, lda, x, type, workspace, workspace_size);
#else
    detail::solver_scalar_f32(m, pivot, lda, x, type, workspace, workspace_size);
#endif
}

//-----------------------------------------------------------------------------
void linear_solver(double* m, linalg_int* pivot, linalg_int lda, double* x, linear_solver_type type,
    void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(m != nullptr && x != nullptr, "linear_solver: m and x must not be null");
    LOGGING_CHECK(lda > 0, "linear_solver: lda must be positive", lda);
#if LINALG_HAS_MKL
    detail::solver_mkl_f64(m, pivot, lda, x, type, workspace, workspace_size);
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    detail::solver_blas_f64(m, pivot, lda, x, type, workspace, workspace_size);
#else
    detail::solver_scalar_f64(m, pivot, lda, x, type, workspace, workspace_size);
#endif
}
}  // namespace linalg
