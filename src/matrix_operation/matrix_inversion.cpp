#include "include/matrix_operation/matrix_inversion.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include <include/logging.h>

#if LINALG_HAS_MKL
#include <mkl.h>

#include "include/common/workspace_arena.h"
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
#include <lapacke.h>

#include "include/common/workspace_arena.h"
#else
#include <algorithm>

#include "include/common/macros.h"
#include "include/common/workspace_arena.h"
#endif

namespace linalg
{
namespace detail
{

// Null checks shared by every invert_backend<T, Backend>::apply below, so
// callers (tests included) get the same validation regardless of which
// backend is active. There is no non-detail matrix_invert() wrapper to put
// this in any more — callers use invert_backend<T, kActiveBackend>::apply
// directly.
void check_invert_inputs(const void* m, linalg_int lda)
{
    LOGGING_CHECK(m != nullptr, "invert_backend::apply: m must not be null");
    LOGGING_CHECK(lda > 0, "invert_backend::apply: lda must be positive", lda);
}

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that determines linalg::kActiveBackend
// (include/common/backend.h). Each `apply` below is an explicit
// specialization of the one member matrix_inversion.h declares (not of the
// whole class), so it's defined exactly once, here, for whichever (T,
// Backend) pair is active — mkl.h / lapacke.h stay out of the public
// header. Other translation units only ever see the header's declaration
// and link against this definition; they never instantiate a body of their
// own, so there's nothing to silently get wrong by omission. No GPU
// implementation exists for this op.
#if LINALG_HAS_MKL

namespace
{
// getri needs its own LAPACK work array (unlike getrf/potrf/potri, which
// need none); lwork is queried with a dummy lwork=-1 call per LAPACK
// convention (the optimal size comes back in work[0]).
template <typename T, typename GetriWorkFn>
lapack_int getri_lwork(GetriWorkFn getri_work, lapack_int n)
{
    T          query[1];
    getri_work(LAPACK_ROW_MAJOR, n, nullptr, n, nullptr, query, -1);
    return static_cast<lapack_int>(query[0]);
}

// Layout: lu_decomposition's own nested workspace (0 unless LU_LINEAR_SOLVER
// needs to factor), then getri's work array (needed for both LU types).
template <typename T, typename GetriWorkFn>
size_t invert_mkl_workspace_size_impl(GetriWorkFn getri_work, linalg_int lda, linear_solver_type type)
{
    detail::workspace_arena arena;
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        arena.next<unsigned char>(lu_decomposition_workspace_size(static_cast<const T*>(nullptr), lda));
        arena.next<T>(static_cast<size_t>(getri_lwork<T>(getri_work, static_cast<lapack_int>(lda))));
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        arena.next<T>(static_cast<size_t>(getri_lwork<T>(getri_work, static_cast<lapack_int>(lda))));
        break;
    default:
        break;
    }
    return arena.total_size();
}
}  // namespace

size_t invert_mkl_workspace_size_f32(linalg_int lda, linear_solver_type type)
{
    return invert_mkl_workspace_size_impl<float>(LAPACKE_sgetri_work, lda, type);
}

size_t invert_mkl_workspace_size_f64(linalg_int lda, linear_solver_type type)
{
    return invert_mkl_workspace_size_impl<double>(LAPACKE_dgetri_work, lda, type);
}

template <>
void invert_backend<float, linalg_backend::mkl>::apply(float* m, linalg_int* pivot, linalg_int lda,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    check_invert_inputs(m, lda);
    LOGGING_CHECK(workspace_size >= invert_mkl_workspace_size_f32(lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws = arena.next<unsigned char>(
            lu_decomposition_workspace_size(static_cast<const float*>(nullptr), lda));
        const auto lwork = getri_lwork<float>(LAPACKE_sgetri_work, n);
        auto*      work  = arena.next<float>(static_cast<size_t>(lwork));
        if (lu_backend<float, kActiveBackend>::apply(m, lda, pivot, lu_ws,
                lu_decomposition_workspace_size(static_cast<const float*>(nullptr), lda)))
        {
            LAPACKE_sgetri_work(LAPACK_ROW_MAJOR, n, m, n, pivot, work, lwork);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_backend<float, kActiveBackend>::apply(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
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
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
void invert_backend<double, linalg_backend::mkl>::apply(double* m, linalg_int* pivot, linalg_int lda,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    check_invert_inputs(m, lda);
    LOGGING_CHECK(workspace_size >= invert_mkl_workspace_size_f64(lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws = arena.next<unsigned char>(
            lu_decomposition_workspace_size(static_cast<const double*>(nullptr), lda));
        const auto lwork = getri_lwork<double>(LAPACKE_dgetri_work, n);
        auto*      work  = arena.next<double>(static_cast<size_t>(lwork));
        if (lu_backend<double, kActiveBackend>::apply(m, lda, pivot, lu_ws,
                lu_decomposition_workspace_size(static_cast<const double*>(nullptr), lda)))
        {
            LAPACKE_dgetri_work(LAPACK_ROW_MAJOR, n, m, n, pivot, work, lwork);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_backend<double, kActiveBackend>::apply(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
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
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

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
size_t invert_blas_workspace_size_impl(GetriWorkFn getri_work, linalg_int lda, linear_solver_type type)
{
    detail::workspace_arena arena;
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
        arena.next<unsigned char>(lu_decomposition_workspace_size(static_cast<const T*>(nullptr), lda));
        arena.next<lapack_int>(static_cast<size_t>(lda));
        arena.next<T>(static_cast<size_t>(getri_lwork<T>(getri_work, static_cast<lapack_int>(lda))));
        break;
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        arena.next<lapack_int>(static_cast<size_t>(lda));
        arena.next<T>(static_cast<size_t>(getri_lwork<T>(getri_work, static_cast<lapack_int>(lda))));
        break;
    default:
        break;
    }
    return arena.total_size();
}
}  // namespace

size_t invert_blas_workspace_size_f32(linalg_int lda, linear_solver_type type)
{
    return invert_blas_workspace_size_impl<float>(LAPACKE_sgetri_work, lda, type);
}

size_t invert_blas_workspace_size_f64(linalg_int lda, linear_solver_type type)
{
    return invert_blas_workspace_size_impl<double>(LAPACKE_dgetri_work, lda, type);
}

template <>
void invert_backend<float, linalg_backend::blas_lapacke>::apply(float* m, linalg_int* pivot, linalg_int lda,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    check_invert_inputs(m, lda);
    LOGGING_CHECK(workspace_size >= invert_blas_workspace_size_f32(lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws = arena.next<unsigned char>(
            lu_decomposition_workspace_size(static_cast<const float*>(nullptr), lda));
        auto*      ipiv  = arena.next<lapack_int>(static_cast<size_t>(lda));
        const auto lwork = getri_lwork<float>(LAPACKE_sgetri_work, n);
        auto*      work  = arena.next<float>(static_cast<size_t>(lwork));
        if (lu_backend<float, kActiveBackend>::apply(m, lda, pivot, lu_ws,
                lu_decomposition_workspace_size(static_cast<const float*>(nullptr), lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_sgetri_work(LAPACK_ROW_MAJOR, n, m, n, ipiv, work, lwork);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_backend<float, kActiveBackend>::apply(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
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
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
void invert_backend<double, linalg_backend::blas_lapacke>::apply(double* m, linalg_int* pivot, linalg_int lda,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    check_invert_inputs(m, lda);
    LOGGING_CHECK(workspace_size >= invert_blas_workspace_size_f64(lda, type),
        "matrix_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              n = static_cast<lapack_int>(lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        auto* lu_ws = arena.next<unsigned char>(
            lu_decomposition_workspace_size(static_cast<const double*>(nullptr), lda));
        auto*      ipiv  = arena.next<lapack_int>(static_cast<size_t>(lda));
        const auto lwork = getri_lwork<double>(LAPACKE_dgetri_work, n);
        auto*      work  = arena.next<double>(static_cast<size_t>(lwork));
        if (lu_backend<double, kActiveBackend>::apply(m, lda, pivot, lu_ws,
                lu_decomposition_workspace_size(static_cast<const double*>(nullptr), lda)))
        {
            to_lapack_pivots(pivot, lda, ipiv);
            LAPACKE_dgetri_work(LAPACK_ROW_MAJOR, n, m, n, ipiv, work, lwork);
        }
        break;
    }
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        if (cholesky_backend<double, kActiveBackend>::apply(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
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
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

#else

template <typename T>
size_t invert_scalar_workspace_size(linalg_int lda, linear_solver_type type)
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
void lu_invert(T* m, LINALG_UNUSED const linalg_int* pivot, linalg_int lda, double* inv_m, double* work)
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
void matrix_invert_scalar_impl(T* m, linalg_int* pivot, linalg_int lda, linear_solver_type type,
    void* workspace, size_t workspace_size)
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
            cholesky_backend<T, kActiveBackend>::apply(m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
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
void invert_backend<float, linalg_backend::native>::apply(float* m, linalg_int* pivot, linalg_int lda,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    check_invert_inputs(m, lda);
    matrix_invert_scalar_impl(m, pivot, lda, type, workspace, workspace_size);
}

template <>
void invert_backend<double, linalg_backend::native>::apply(double* m, linalg_int* pivot, linalg_int lda,
    linear_solver_type type, void* workspace, size_t workspace_size)
{
    check_invert_inputs(m, lda);
    matrix_invert_scalar_impl(m, pivot, lda, type, workspace, workspace_size);
}

#endif

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
        return lu_decomposition_workspace_size(static_cast<const T*>(nullptr), lda);
    default:
        return 0;
    }
}

template <typename T>
T matrix_determinant_helper(
    T* m, linalg_int* pivot, linalg_int lda, linear_solver_type type, void* workspace, size_t workspace_size)
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
        if (cholesky_backend<T, kActiveBackend>::apply(m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
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

}  // namespace detail

size_t matrix_invert_workspace_size(const float* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
#if LINALG_HAS_MKL
    return detail::invert_mkl_workspace_size_f32(lda, type);
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    return detail::invert_blas_workspace_size_f32(lda, type);
#else
    return detail::invert_scalar_workspace_size<float>(lda, type);
#endif
}

size_t matrix_invert_workspace_size(const double* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
#if LINALG_HAS_MKL
    return detail::invert_mkl_workspace_size_f64(lda, type);
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    return detail::invert_blas_workspace_size_f64(lda, type);
#else
    return detail::invert_scalar_workspace_size<double>(lda, type);
#endif
}

size_t matrix_determinant_workspace_size(const float* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
    return detail::matrix_determinant_workspace_size_helper<float>(lda, type);
}

size_t matrix_determinant_workspace_size(const double* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
    return detail::matrix_determinant_workspace_size_helper<double>(lda, type);
}

//-----------------------------------------------------------------------------
float matrix_determinant(float* m, linalg_int* pivot, linalg_int lda, linear_solver_type type,
    void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(m != nullptr, "matrix_determinant: m must not be null");
    LOGGING_CHECK(lda > 0, "matrix_determinant: lda must be positive", lda);
    return detail::matrix_determinant_helper(m, pivot, lda, type, workspace, workspace_size);
}

//-----------------------------------------------------------------------------
double matrix_determinant(double* m, linalg_int* pivot, linalg_int lda, linear_solver_type type,
    void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(m != nullptr, "matrix_determinant: m must not be null");
    LOGGING_CHECK(lda > 0, "matrix_determinant: lda must be positive", lda);
    return detail::matrix_determinant_helper(m, pivot, lda, type, workspace, workspace_size);
}
}  // namespace linalg
