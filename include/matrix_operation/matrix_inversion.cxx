#include "include/matrix_operation/matrix_inversion.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/util/exception.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
#include <lapacke.h>

#include <vector>
#else
#include <algorithm>
#include <vector>

#include "include/common/macros.h"
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// matrix_invert() overloads below call — exactly one is ever built. No GPU
// implementation exists for this op.
#if defined(LINALG_ENABLE_MKL)

void invert_mkl_f32(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                LAPACKE_sgetri(LAPACK_ROW_MAJOR, lda, m, lda, pivot);
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            LAPACKE_sgetri(LAPACK_ROW_MAJOR, lda, m, lda, pivot);
            break;
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

void invert_mkl_f64(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                LAPACKE_dgetri(LAPACK_ROW_MAJOR, lda, m, lda, pivot);
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, lda, m, lda, pivot);
            break;
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', lda, m, lda);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

namespace
{
// See lu_decomposition.cxx's blas_lapack branch.
std::vector<lapack_int> to_lapack_pivots(const quarisma_int* pivot, quarisma_int n)
{
    std::vector<lapack_int> ipiv(static_cast<size_t>(n));
    for (quarisma_int i = 0; i < n; ++i)
    {
        ipiv[static_cast<size_t>(i)] = static_cast<lapack_int>(pivot[i]);
    }
    return ipiv;
}
}  // namespace

void invert_blas_f32(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    const auto n = static_cast<lapack_int>(lda);
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                auto ipiv = to_lapack_pivots(pivot, lda);
                LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, m, n, ipiv.data());
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        {
            auto ipiv = to_lapack_pivots(pivot, lda);
            LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, m, n, ipiv.data());
            break;
        }
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

void invert_blas_f64(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    const auto n = static_cast<lapack_int>(lda);
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
            if (lu_decomposition(m, lda, pivot))
            {
                auto ipiv = to_lapack_pivots(pivot, lda);
                LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, m, n, ipiv.data());
            }
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
            if (cholesky_decomposition(m, lda, cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
            }
            break;
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        {
            auto ipiv = to_lapack_pivots(pivot, lda);
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, m, n, ipiv.data());
            break;
        }
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, m, n);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

#else

namespace
{

#define A(i, j) m[(i) * lda + (j)]
#define IA(i, j) inv_m[(i) * lda + (j)]

template <typename T>
void lu_invert(T* m, LINALG_UNUSED const quarisma_int* pivot, quarisma_int lda)
{
    using size_type = quarisma_int;
    using value_t   = T;
    std::vector<double> inv_m(lda * lda);
    std::vector<double> work(lda);
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
#ifdef LINALG_LU_PIVOTING
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

template <typename T>
void cholesky_invert(T* m, quarisma_int lda)
{
    using value_t   = T;
    using size_type = quarisma_int;

    std::vector<T> work(lda);

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
void matrix_invert_scalar_impl(T* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        {
            if (lu_decomposition(m, lda, pivot))
            {
                lu_invert(m, pivot, lda);
            }
        }
        break;

        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            lu_invert(m, pivot, lda);
            break;

        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        {
            if (cholesky_decomposition(m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                cholesky_invert(m, lda);
            }
        }
        break;

        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            cholesky_invert(m, lda);
            break;
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

}  // namespace

void invert_scalar_f32(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    matrix_invert_scalar_impl(m, pivot, lda, type);
}

void invert_scalar_f64(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    matrix_invert_scalar_impl(m, pivot, lda, type);
}

#endif

// matrix_determinant just reads the diagonal of an already-decomposed
// buffer — there is no vendor-library "determinant" routine to call into,
// so this always runs directly against whatever lu_decomposition /
// cholesky_decomposition already wrote into `m`.
#define A(i, j) m[(i) * lda + (j)]

template <typename T>
T lu_determinant(T* m, quarisma_int lda)
{
    T det = m[0];
    for (quarisma_int i = 1; i < lda; ++i)
    {
        det *= A(i, i);
    }
    return det;
}

template <typename T>
T cholesky_determinant(T* m, quarisma_int lda)
{
    T det = m[0];
    for (quarisma_int i = 1; i < lda; ++i)
    {
        det *= A(i, i);
    }
    return det;
}

#undef A

template <typename T>
T matrix_determinant_helper(T* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    T ret = 0.;
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        {
            if (lu_decomposition(m, lda, pivot))
            {
                ret = lu_determinant(m, lda);
            }
        }
            return ret;

        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            return lu_determinant(m, lda);

        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        {
            if (cholesky_decomposition(m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                ret = cholesky_determinant(m, lda);
            }
        }
            return ret;

        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            return cholesky_determinant(m, lda);
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

}  // namespace detail

//-----------------------------------------------------------------------------
void matrix_invert(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
#if defined(LINALG_ENABLE_MKL)
    detail::invert_mkl_f32(m, pivot, lda, type);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::invert_blas_f32(m, pivot, lda, type);
#else
    detail::invert_scalar_f32(m, pivot, lda, type);
#endif
}

//-----------------------------------------------------------------------------
void matrix_invert(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
#if defined(LINALG_ENABLE_MKL)
    detail::invert_mkl_f64(m, pivot, lda, type);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::invert_blas_f64(m, pivot, lda, type);
#else
    detail::invert_scalar_f64(m, pivot, lda, type);
#endif
}

//-----------------------------------------------------------------------------
float matrix_determinant(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    return detail::matrix_determinant_helper(m, pivot, lda, type);
}

//-----------------------------------------------------------------------------
double matrix_determinant(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    return detail::matrix_determinant_helper(m, pivot, lda, type);
}
}  // namespace linalg
