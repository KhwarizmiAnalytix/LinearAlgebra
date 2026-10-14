#include "include/matrix_operation/lu_decomposition.h"

#include <include/logging.h>

#if defined(LINALG_ENABLE_MKL)
#include <mkl_lapacke.h>
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
// LAPACKE-dependent — see CMakeLists.txt and
// cholesky_decomposition.cpp: Apple's Accelerate framework has no LAPACKE
// wrapper, so this branch only compiles when real lapacke.h exists;
// otherwise the #else below falls through to the scalar implementation.
#include <lapacke.h>

#include "include/common/workspace_arena.h"
#else
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "include/common/macros.h"
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// lu_decomposition() overloads below call — exactly one is ever built.
#if defined(LINALG_ENABLE_MKL)

// No scratch needed: an ILP64 MKL build has MKL_INT == linalg_int, so the
// caller's pivot buffer is passed straight through to getrf.
bool lu_mkl_f32(float* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/)
{
    return LAPACKE_sgetrf(LAPACK_ROW_MAJOR, lda, lda, m, lda, pivot) == 0;
}

bool lu_mkl_f64(double* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/)
{
    return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, lda, lda, m, lda, pivot) == 0;
}

#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

namespace
{
// lapack_int (a generic LAPACKE's pivot element type, typically 32-bit even
// on 64-bit platforms) does not necessarily match linalg_int (64-bit), so
// the caller's pivot buffer can't be passed straight through — unlike the
// mkl backend, which assumes an ILP64 MKL build where MKL_INT == linalg_int.
// `ipiv` (n lapack_int) is caller-owned scratch, carved from `workspace`.
size_t lu_blas_workspace_size(linalg_int lda)
{
    detail::workspace_arena arena;
    arena.next<lapack_int>(static_cast<size_t>(lda));
    return arena.total_size();
}

bool lu_blas_impl(lapack_int (*getrf)(int, lapack_int, lapack_int, void*, lapack_int, lapack_int*),
    void*         m,
    linalg_int  lda,
    linalg_int* pivot,
    void*       workspace,
    size_t      workspace_size)
{
    LOGGING_CHECK(workspace_size >= lu_blas_workspace_size(lda), "lu_decomposition: workspace_size too small");
    const auto              n = static_cast<lapack_int>(lda);
    detail::workspace_arena arena(workspace, workspace_size);
    lapack_int*              ipiv = arena.next<lapack_int>(static_cast<size_t>(n));
    const auto               info = getrf(LAPACK_ROW_MAJOR, n, n, m, n, ipiv);
    for (lapack_int i = 0; i < n; ++i)
    {
        pivot[i] = static_cast<linalg_int>(ipiv[i]);
    }
    return info == 0;
}
}  // namespace

size_t lu_blas_workspace_size_f32(linalg_int lda)
{
    return lu_blas_workspace_size(lda);
}

size_t lu_blas_workspace_size_f64(linalg_int lda)
{
    return lu_blas_workspace_size(lda);
}

bool lu_blas_f32(float* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size)
{
    return lu_blas_impl(
        [](int order, lapack_int r, lapack_int c, void* a, lapack_int ld, lapack_int* ip)
        { return LAPACKE_sgetrf(order, r, c, static_cast<float*>(a), ld, ip); },
        m,
        lda,
        pivot,
        workspace,
        workspace_size);
}

bool lu_blas_f64(double* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size)
{
    return lu_blas_impl(
        [](int order, lapack_int r, lapack_int c, void* a, lapack_int ld, lapack_int* ip)
        { return LAPACKE_dgetrf(order, r, c, static_cast<double*>(a), ld, ip); },
        m,
        lda,
        pivot,
        workspace,
        workspace_size);
}

#else

namespace
{

#define A(i, j) m[(i) * lda + (j)]

template <typename T>
void lu_decomposition_scalar_impl(
    T* m, linalg_long lda, LINALG_UNUSED linalg_int* pivot, LINALG_UNUSED T tolerance)
{
    using value_t   = T;
    using size_type = linalg_long;

#ifdef LINALG_LU_PIVOTING
    // Initialize pivot array. `pivot` holds exactly `lda` entries (LAPACK's
    // ipiv convention: pivot[i] is the 1-based row swapped with row i) —
    // pivotCount (the swap count, used only to flip the determinant's sign
    // under odd permutations) is tracked locally rather than as a trailing
    // pivot[lda], which used to overflow the caller's n-sized buffer by one
    // element (see docs/backend_roadmap.md §2.1 item 1).
    for (size_type i = 0; i < lda; ++i)
    {
        pivot[i] = static_cast<linalg_int>(i + 1);
    }
    linalg_int pivotCount = 0;
#endif
    for (size_type i = 0; i < lda; ++i)
    {
#ifdef LINALG_LU_PIVOTING
        value_t   maxA = std::abs(A(i, i));
        size_type imax = i;
        for (size_type k = i + 1; k < lda; ++k)
        {
            const auto absA = std::fabs(A(k, i));
            if (absA > maxA)
            {
                maxA = absA;
                imax = k;
            }
        }
        if (maxA < tolerance)
        {
            return false;
        }
        if (imax != i)
        {
            pivot[i] = pivot[imax];
            for (size_type j = 0; j < lda; ++j)
            {
                auto& ptri = A(i, j);
                auto& ptrj = A(imax, j);
                std::swap(ptri, ptrj);
            }
            pivotCount++;
        }
#endif
        auto norm = 1. / A(i, i);
        auto size = lda - i - 1;

        const auto* const l_i = &A(i, i + 1);
        for (size_type j = i + 1; j < lda; ++j)
        {
            auto* l_j = &A(j, i);
            l_j[0] *= static_cast<T>(norm);
            const auto a_ji = l_j[0];
            l_j++;
            for (size_type k = 0; k < size; ++k)
            {
                l_j[k] -= a_ji * l_i[k];
            }
        }
    }
}

#undef A

}  // namespace

void lu_scalar_f32(float* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/)
{
    lu_decomposition_scalar_impl(m, lda, pivot, std::numeric_limits<float>::epsilon());
}

void lu_scalar_f64(double* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/)
{
    lu_decomposition_scalar_impl(m, lda, pivot, std::numeric_limits<double>::epsilon());
}

#endif

}  // namespace detail

size_t lu_decomposition_workspace_size(const float* /*type_tag*/, linalg_int lda)
{
#if defined(LINALG_ENABLE_MKL)
    (void)lda;
    return 0;
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::lu_blas_workspace_size_f32(lda);
#else
    (void)lda;
    return 0;
#endif
}

size_t lu_decomposition_workspace_size(const double* /*type_tag*/, linalg_int lda)
{
#if defined(LINALG_ENABLE_MKL)
    (void)lda;
    return 0;
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::lu_blas_workspace_size_f64(lda);
#else
    (void)lda;
    return 0;
#endif
}

bool lu_decomposition(float* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(m != nullptr && pivot != nullptr, "lu_decomposition: m and pivot must not be null");
    LOGGING_CHECK(lda > 0, "lu_decomposition: lda must be positive", lda);
#if defined(LINALG_ENABLE_MKL)
    return detail::lu_mkl_f32(m, lda, pivot, workspace, workspace_size);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::lu_blas_f32(m, lda, pivot, workspace, workspace_size);
#else
    detail::lu_scalar_f32(m, lda, pivot, workspace, workspace_size);
    return true;
#endif
}

bool lu_decomposition(double* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(m != nullptr && pivot != nullptr, "lu_decomposition: m and pivot must not be null");
    LOGGING_CHECK(lda > 0, "lu_decomposition: lda must be positive", lda);
#if defined(LINALG_ENABLE_MKL)
    return detail::lu_mkl_f64(m, lda, pivot, workspace, workspace_size);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::lu_blas_f64(m, lda, pivot, workspace, workspace_size);
#else
    detail::lu_scalar_f64(m, lda, pivot, workspace, workspace_size);
    return true;
#endif
}

}  // namespace linalg
