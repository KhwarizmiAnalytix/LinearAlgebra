#include "include/matrix_operation/lu_decomposition.h"

#include "include/util/exception.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl_lapacke.h>
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
// LAPACKE-dependent — see CMakeLists.txt and
// cholesky_decomposition.cxx: Apple's Accelerate framework has no LAPACKE
// wrapper, so this branch only compiles when real lapacke.h exists;
// otherwise the #else below falls through to the scalar implementation.
#include <lapacke.h>

#include <vector>
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

bool lu_mkl_f32(float* m, linalg_int lda, linalg_int* pivot)
{
    return LAPACKE_sgetrf(LAPACK_ROW_MAJOR, lda, lda, m, lda, pivot) == 0;
}

bool lu_mkl_f64(double* m, linalg_int lda, linalg_int* pivot)
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
bool lu_blas_impl(lapack_int (*getrf)(int, lapack_int, lapack_int, void*, lapack_int, lapack_int*),
    void*         m,
    linalg_int  lda,
    linalg_int* pivot)
{
    const auto              n = static_cast<lapack_int>(lda);
    std::vector<lapack_int> ipiv(static_cast<size_t>(n));
    const auto              info = getrf(LAPACK_ROW_MAJOR, n, n, m, n, ipiv.data());
    for (lapack_int i = 0; i < n; ++i)
    {
        pivot[i] = static_cast<linalg_int>(ipiv[static_cast<size_t>(i)]);
    }
    return info == 0;
}
}  // namespace

bool lu_blas_f32(float* m, linalg_int lda, linalg_int* pivot)
{
    return lu_blas_impl(
        [](int order, lapack_int r, lapack_int c, void* a, lapack_int ld, lapack_int* ip)
        { return LAPACKE_sgetrf(order, r, c, static_cast<float*>(a), ld, ip); },
        m,
        lda,
        pivot);
}

bool lu_blas_f64(double* m, linalg_int lda, linalg_int* pivot)
{
    return lu_blas_impl(
        [](int order, lapack_int r, lapack_int c, void* a, lapack_int ld, lapack_int* ip)
        { return LAPACKE_dgetrf(order, r, c, static_cast<double*>(a), ld, ip); },
        m,
        lda,
        pivot);
}

#else

namespace
{

#define A(i, j) m[(i) * lda + (j)]

template <typename T>
bool lu_decomposition_scalar_impl(
    T* m, linalg_long lda, LINALG_UNUSED linalg_int* pivot, LINALG_UNUSED T tolerance)
{
    using value_t   = T;
    using size_type = linalg_long;

#ifdef LINALG_LU_PIVOTING
    // Initialize pivot array
    for (size_type i = 0; i <= lda; ++i)
    {
        pivot[i] = static_cast<linalg_int>(i + 1);
    }
    linalg_int& pivotCount = pivot[lda];
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
    return true;
}

#undef A

}  // namespace

bool lu_scalar_f32(float* m, linalg_int lda, linalg_int* pivot)
{
    return lu_decomposition_scalar_impl(m, lda, pivot, std::numeric_limits<float>::epsilon());
}

bool lu_scalar_f64(double* m, linalg_int lda, linalg_int* pivot)
{
    return lu_decomposition_scalar_impl(m, lda, pivot, std::numeric_limits<double>::epsilon());
}

#endif

}  // namespace detail

bool lu_decomposition(float* m, linalg_int lda, linalg_int* pivot)
{
    if (m == nullptr || pivot == nullptr)
    {
        LINALG_THROW("lu_decomposition: m and pivot must not be null");
    }
    if (lda <= 0)
    {
        LINALG_THROW("lu_decomposition: lda must be positive", lda);
    }
#if defined(LINALG_ENABLE_MKL)
    return detail::lu_mkl_f32(m, lda, pivot);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::lu_blas_f32(m, lda, pivot);
#else
    return detail::lu_scalar_f32(m, lda, pivot);
#endif
}

bool lu_decomposition(double* m, linalg_int lda, linalg_int* pivot)
{
    if (m == nullptr || pivot == nullptr)
    {
        LINALG_THROW("lu_decomposition: m and pivot must not be null");
    }
    if (lda <= 0)
    {
        LINALG_THROW("lu_decomposition: lda must be positive", lda);
    }
#if defined(LINALG_ENABLE_MKL)
    return detail::lu_mkl_f64(m, lda, pivot);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    return detail::lu_blas_f64(m, lda, pivot);
#else
    return detail::lu_scalar_f64(m, lda, pivot);
#endif
}

}  // namespace linalg
