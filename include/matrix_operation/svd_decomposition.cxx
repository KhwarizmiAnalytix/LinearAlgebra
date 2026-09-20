#include "include/matrix_operation/svd_decomposition.h"

#include "include/common/configure.h"  // IWYU pragma: keep
#include "include/util/exception.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>

#include <algorithm>

#include "include/memory/allocator.h"
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
#include <lapacke.h>

#include <algorithm>

#include "include/memory/allocator.h"
#else
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "include/memory/allocator.h"
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// svd_decomposition() overloads below call — exactly one is ever built. No
// GPU implementation exists for this op.
#if defined(LINALG_ENABLE_MKL)

void svd_mkl_f32(
    quarisma_long rows,
    quarisma_long columns,
    float*      A,
    quarisma_long lda,
    float*      S,
    float*      U,
    quarisma_long ldu,
    float*      VT,
    quarisma_long ldv)
{
    using Allocator = linalg::allocator<float>;
    auto* temp      = Allocator::allocate(ldu - 1);
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_sgesvd(
            LAPACK_ROW_MAJOR, 'O', 'S', rows, columns, U, ldu, S, U, ldu, VT, ldv, temp);  //NOLINT
        if (info != 0)
        {
            LINALG_THROW("mkl SVD was unsuccessful");
        }
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_sgesvd(
            LAPACK_ROW_MAJOR, 'S', 'O', rows, columns, VT, ldv, S, U, ldu, VT, ldv, temp);  //NOLINT
        if (info != 0)
        {
            LINALG_THROW("mkl SVD was unsuccessful");
        }
    }
    Allocator::free(temp);
}

void svd_mkl_f64(
    quarisma_long rows,
    quarisma_long columns,
    double*     A,
    quarisma_long lda,
    double*     S,
    double*     U,
    quarisma_long ldu,
    double*     VT,
    quarisma_long ldv)
{
    using Allocator = linalg::allocator<double>;
    auto* temp      = Allocator::allocate(ldu - 1);
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'O', 'S', rows, columns, U, ldu, S, U, ldu, VT, ldv, temp);  //NOLINT
        if (info != 0)
        {
            LINALG_THROW("mkl SVD was unsuccessful");
        }
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'S', 'O', rows, columns, VT, ldv, S, U, ldu, VT, ldv, temp);  //NOLINT
        if (info != 0)
        {
            LINALG_THROW("mkl SVD was unsuccessful");
        }
    }
    Allocator::free(temp);
}

#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

void svd_blas_f32(
    quarisma_long rows,
    quarisma_long columns,
    float*      A,
    quarisma_long lda,
    float*      S,
    float*      U,
    quarisma_long ldu,
    float*      VT,
    quarisma_long ldv)
{
    using Allocator = linalg::allocator<float>;
    auto* temp      = Allocator::allocate(ldu - 1);
    const auto r = static_cast<lapack_int>(rows);
    const auto c = static_cast<lapack_int>(columns);
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_sgesvd(
            LAPACK_ROW_MAJOR, 'O', 'S', r, c, U, static_cast<lapack_int>(ldu), S, U,
            static_cast<lapack_int>(ldu), VT, static_cast<lapack_int>(ldv), temp);
        if (info != 0)
        {
            LINALG_THROW("blas_lapack SVD was unsuccessful");
        }
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_sgesvd(
            LAPACK_ROW_MAJOR, 'S', 'O', r, c, VT, static_cast<lapack_int>(ldv), S, U,
            static_cast<lapack_int>(ldu), VT, static_cast<lapack_int>(ldv), temp);
        if (info != 0)
        {
            LINALG_THROW("blas_lapack SVD was unsuccessful");
        }
    }
    Allocator::free(temp);
}

void svd_blas_f64(
    quarisma_long rows,
    quarisma_long columns,
    double*     A,
    quarisma_long lda,
    double*     S,
    double*     U,
    quarisma_long ldu,
    double*     VT,
    quarisma_long ldv)
{
    using Allocator = linalg::allocator<double>;
    auto* temp      = Allocator::allocate(ldu - 1);
    const auto r = static_cast<lapack_int>(rows);
    const auto c = static_cast<lapack_int>(columns);
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'O', 'S', r, c, U, static_cast<lapack_int>(ldu), S, U,
            static_cast<lapack_int>(ldu), VT, static_cast<lapack_int>(ldv), temp);
        if (info != 0)
        {
            LINALG_THROW("blas_lapack SVD was unsuccessful");
        }
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'S', 'O', r, c, VT, static_cast<lapack_int>(ldv), S, U,
            static_cast<lapack_int>(ldu), VT, static_cast<lapack_int>(ldv), temp);
        if (info != 0)
        {
            LINALG_THROW("blas_lapack SVD was unsuccessful");
        }
    }
    Allocator::free(temp);
}

#else

namespace
{

#define U(i, j) U_[(i) * dim[0] + (j)]
#define S(i, j) S_[(i) * dim[1] + (j)]
#define V(i, j) V_[(i) * dim[1] + (j)]

template <class T>
void GivensL(T* S_, const quarisma_long dim[2], quarisma_long m, T a, T b)  // NOLINT
{
    T r = sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

#pragma omp parallel for
    for (quarisma_int i = 0; i < static_cast<quarisma_int>(dim[1]); i++)
    {
        T S0 = S(m + 0, i);
        T S1 = S(m + 1, i);
        S(m, i) += S0 * (c - 1);
        S(m, i) += S1 * (-s);

        S(m + 1, i) += S0 * (s);
        S(m + 1, i) += S1 * (c - 1);
    }
}

template <class T>
void GivensR(T* S_, const quarisma_long dim[2], quarisma_long m, T a, T b)  // NOLINT
{
    T r = sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

#pragma omp parallel for
    for (quarisma_int i = 0; i < static_cast<quarisma_int>(dim[0]); i++)
    {
        T S0 = S(i, m + 0);
        T S1 = S(i, m + 1);
        S(i, m) += S0 * (c - 1);
        S(i, m) += S1 * (-s);

        S(i, m + 1) += S0 * (s);
        S(i, m + 1) += S1 * (c - 1);
    }
}

template <class T>
void SVD(const quarisma_long dim[2], T* U_, T* S_, T* V_, T eps = -1)  // NOLINT
{
    assert(dim[0] >= dim[1]);

    {  // Bi-diagonalization
        const auto     n = std::min(dim[0], dim[1]);
        std::vector<T> house_vec(std::max(dim[0], dim[1]));
        for (quarisma_long i = 0; i < n; i++)
        {
            // Column Householder
            {
                T x1 = S(i, i);
                if (x1 < 0)
                {
                    x1 = -x1;
                }

                T x_inv_norm = 0;
                for (quarisma_long j = i; j < dim[0]; j++)
                {
                    x_inv_norm += S(j, i) * S(j, i);
                }
                if (x_inv_norm > 0)
                {
                    x_inv_norm = 1 / sqrt(x_inv_norm);
                }

                T alpha = sqrt(1 + x1 * x_inv_norm);
                T beta  = x_inv_norm / alpha;

                house_vec[i] = -alpha;
                for (quarisma_long j = i + 1; j < dim[0]; j++)
                {
                    house_vec[j] = -beta * S(j, i);
                }
                if (S(i, i) < 0)
                {
                    for (quarisma_long j = i + 1; j < dim[0]; j++)
                    {
                        house_vec[j] = -house_vec[j];
                    }
                }
            }
#pragma omp parallel for
            for (auto k = static_cast<quarisma_int>(i); k < static_cast<quarisma_int>(dim[1]); k++)
            {
                T dot_prod = 0;
                for (quarisma_long j = i; j < dim[0]; j++)
                {
                    dot_prod += S(j, k) * house_vec[j];
                }
                for (quarisma_long j = i; j < dim[0]; j++)
                {
                    S(j, k) -= dot_prod * house_vec[j];
                }
            }
#pragma omp parallel for
            for (quarisma_int k = 0; k < static_cast<quarisma_int>(dim[0]); k++)
            {
                T dot_prod = 0;
                for (quarisma_long j = i; j < dim[0]; j++)
                {
                    dot_prod += U(k, j) * house_vec[j];
                }
                for (quarisma_long j = i; j < dim[0]; j++)
                {
                    U(k, j) -= dot_prod * house_vec[j];
                }
            }

            // Row Householder
            if (i >= n - 1)
            {
                continue;
            }
            {
                T x1 = S(i, i + 1);
                if (x1 < 0)
                {
                    x1 = -x1;
                }

                T x_inv_norm = 0;
                for (quarisma_long j = i + 1; j < dim[1]; j++)
                {
                    x_inv_norm += S(i, j) * S(i, j);
                }
                if (x_inv_norm > 0)
                {
                    x_inv_norm = 1 / sqrt(x_inv_norm);
                }

                T alpha = sqrt(1 + x1 * x_inv_norm);
                T beta  = x_inv_norm / alpha;

                house_vec[i + 1] = -alpha;
                for (quarisma_long j = i + 2; j < dim[1]; j++)
                {
                    house_vec[j] = -beta * S(i, j);
                }
                if (S(i, i + 1) < 0)
                {
                    for (quarisma_long j = i + 2; j < dim[1]; j++)
                    {
                        house_vec[j] = -house_vec[j];
                    }
                }
            }
#pragma omp parallel for
            for (auto k = static_cast<quarisma_int>(i); k < static_cast<quarisma_int>(dim[0]); k++)
            {
                T dot_prod = 0;
                for (quarisma_long j = i + 1; j < dim[1]; j++)
                {
                    dot_prod += S(k, j) * house_vec[j];
                }
                for (quarisma_long j = i + 1; j < dim[1]; j++)
                {
                    S(k, j) -= dot_prod * house_vec[j];
                }
            }
#pragma omp parallel for
            for (quarisma_int k = 0; k < static_cast<quarisma_int>(dim[1]); k++)
            {
                T dot_prod = 0;
                for (quarisma_long j = i + 1; j < dim[1]; j++)
                {
                    dot_prod += V(j, k) * house_vec[j];
                }
                for (quarisma_long j = i + 1; j < dim[1]; j++)
                {
                    V(j, k) -= dot_prod * house_vec[j];
                }
            }
        }
    }

    quarisma_long k0 = 0;
    if (eps < 0)
    {
        eps = 1.0;
        while (eps + (T)1.0 > 1.0)
        {
            eps *= 0.5;
        }
        eps *= 64.0;
    }
    while (k0 < dim[1] - 1)
    {  // Diagonalization
        T S_max = 0.0;
        for (quarisma_long i = 0; i < dim[1]; i++)
        {
            S_max = (S_max > S(i, i) ? S_max : S(i, i));
        }

        while (k0 < dim[1] - 1 && fabs(S(k0, k0 + 1)) <= eps * S_max)
        {
            k0++;
        }
        if (k0 == dim[1] - 1)
        {
            continue;
        }

        quarisma_long n = k0 + 2;
        while (n < dim[1] && fabs(S(n - 1, n)) > eps * S_max)
        {
            n++;
        }

        T alpha = 0;
        T beta  = 0;
        {
            // Compute mu
            T C[2][2];  // NOLINT

            auto s_n2_2 = S(n - 2, n - 2);
            auto s_n2_1 = S(n - 2, n - 1);
            auto s_n1_1 = S(n - 1, n - 1);

            C[0][0] = s_n2_2 * s_n2_2;
            if (n - k0 > 2)
            {
                auto s_n3_2 = S(n - 3, n - 2);
                C[0][0] += s_n3_2 * s_n3_2;
            }
            C[0][1] = s_n2_2 * s_n2_1;
            C[1][0] = s_n2_2 * s_n2_1;
            C[1][1] = s_n1_1 * s_n1_1 + s_n2_1 * s_n2_1;

            T b = -(C[0][0] + C[1][1]) / 2;
            T c = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            T d = sqrt(b * b - c);

            T lambda1 = -b + d;
            T lambda2 = -b - d;

            T d1 = std::fabs(lambda1 - C[1][1]);
            T d2 = std::fabs(lambda2 - C[1][1]);
            T mu = (d1 < d2 ? lambda1 : lambda2);

            alpha = S(k0, k0) * S(k0, k0) - mu;
            beta  = S(k0, k0) * S(k0, k0 + 1);
        }

        for (quarisma_long k = k0; k < n - 1; k++)
        {
            quarisma_long dimU[2] = {dim[0], dim[0]};  // NOLINT
            quarisma_long dimV[2] = {dim[1], dim[1]};  // NOLINT
            GivensR(S_, dim, k, alpha, beta);
            GivensL(V_, dimV, k, alpha, beta);

            alpha = S(k, k);
            beta  = S(k + 1, k);
            GivensL(S_, dim, k, alpha, beta);
            GivensR(U_, dimU, k, alpha, beta);

            alpha = S(k, k + 1);
            beta  = S(k, k + 2);
        }

        {  // rowsake S bi-diagonal again
            for (quarisma_long i0 = k0; i0 < n - 1; i0++)
            {
                for (quarisma_long i1 = 0; i1 < dim[1]; i1++)
                {
                    if (i0 > i1 || i0 + 1 < i1)
                    {
                        S(i0, i1) = 0;
                    }
                }
            }
            for (quarisma_long i0 = 0; i0 < dim[0]; i0++)
            {
                for (quarisma_long i1 = k0; i1 < n - 1; i1++)
                {
                    if (i0 > i1 || i0 + 1 < i1)
                    {
                        S(i0, i1) = 0;
                    }
                }
            }
            for (quarisma_long i = 0; i < dim[1] - 1; i++)
            {
                if (fabs(S(i, i + 1)) <= eps * S_max)
                {
                    S(i, i + 1) = 0;
                }
            }
        }
    }
}

#undef U
#undef S
#undef V

template <class T, class Allocator = linalg::allocator<T> >
inline void
svd(  //NOLINT
    quarisma_long rows,
    quarisma_long columns,
    T*          A,
    quarisma_long lda,
    T*          S,
    T*          U,
    quarisma_long ldu,
    T*          VT,
    quarisma_long ldv)
{
    const quarisma_long dim[2] = {std::max(rows, columns), std::min(rows, columns)};  // NOLINT

    auto* U_ = Allocator::allocate(dim[0] * dim[0]);
    auto* V_ = Allocator::allocate(dim[1] * dim[1]);
    auto* S_ = Allocator::allocate(rows * columns);

    memset(U_, 0, dim[0] * dim[0] * sizeof(T));
    memset(V_, 0, dim[1] * dim[1] * sizeof(T));

    if (dim[1] == columns)
    {
        for (quarisma_long i = 0; i < dim[0]; i++)
        {
            for (quarisma_long j = 0; j < dim[1]; j++)
            {
                S_[i * dim[1] + j] = A[i * lda + j];
            }
        }
    }
    else
    {
        for (quarisma_long i = 0; i < dim[0]; i++)
        {
            for (quarisma_long j = 0; j < dim[1]; j++)
            {
                S_[i * dim[1] + j] = A[j * lda + i];
            }
        }
    }

    for (quarisma_long i = 0; i < dim[0]; i++)
    {
        U_[i * dim[0] + i] = 1;
    }
    for (quarisma_long i = 0; i < dim[1]; i++)
    {
        V_[i * dim[1] + i] = 1;
    }

    SVD<T>(dim, U_, S_, V_, (T)-1);

    for (quarisma_long i = 0; i < dim[1]; i++)
    {  // Set S
        S[i] = S_[i * dim[1] + i];
    }
    if (dim[1] == columns)
    {  // Set U
        for (quarisma_long i = 0; i < columns; i++)
        {
            for (quarisma_long j = 0; j < columns; j++)
            {
                U[j + ldu * i] = V_[j + i * columns] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
            }
        }

        for (quarisma_long i = 0; i < rows; i++)
        {
            for (quarisma_long j = 0; j < columns; j++)
            {
                VT[j + ldv * i] = U_[j + i * rows];
            }
        }
    }
    else
    {
        for (quarisma_long i = 0; i < rows; i++)
        {
            for (quarisma_long j = 0; j < columns; j++)
            {
                U[j + ldv * i] = U_[i + j * columns] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
            }
        }
        for (quarisma_long i = 0; i < rows; i++)
        {
            for (quarisma_long j = 0; j < rows; j++)
            {
                VT[j + ldu * i] = V_[i + j * rows];
            }
        }
    }

    for (quarisma_long i = 0; i < dim[1]; i++)
    {
        S[i] = S[i] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
    }

    Allocator::free(S_);
    Allocator::free(V_);
    Allocator::free(U_);
}

}  // namespace

void svd_scalar_f32(
    quarisma_long rows,
    quarisma_long columns,
    float*      A,
    quarisma_long lda,
    float*      S,
    float*      U,
    quarisma_long ldu,
    float*      VT,
    quarisma_long ldv)
{
    svd(rows, columns, A, lda, S, VT, ldu, U, ldv);
}

void svd_scalar_f64(
    quarisma_long rows,
    quarisma_long columns,
    double*     A,
    quarisma_long lda,
    double*     S,
    double*     U,
    quarisma_long ldu,
    double*     VT,
    quarisma_long ldv)
{
    svd(rows, columns, A, lda, S, VT, ldu, U, ldv);
}


#endif

}  // namespace detail

//-----------------------------------------------------------------------------
void svd_decomposition(
    quarisma_long rows,
    quarisma_long columns,
    float*      A,
    quarisma_long lda,
    float*      S,
    float*      U,
    quarisma_long ldu,
    float*      VT,
    quarisma_long ldv)
{
#if defined(LINALG_ENABLE_MKL)
    detail::svd_mkl_f32(rows, columns, A, lda, S, U, ldu, VT, ldv);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::svd_blas_f32(rows, columns, A, lda, S, U, ldu, VT, ldv);
#else
    detail::svd_scalar_f32(rows, columns, A, lda, S, U, ldu, VT, ldv);
#endif
}

//-----------------------------------------------------------------------------
void svd_decomposition(
    quarisma_long rows,
    quarisma_long columns,
    double*     A,
    quarisma_long lda,
    double*     S,
    double*     U,
    quarisma_long ldu,
    double*     VT,
    quarisma_long ldv)
{
#if defined(LINALG_ENABLE_MKL)
    detail::svd_mkl_f64(rows, columns, A, lda, S, U, ldu, VT, ldv);
#elif defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)
    detail::svd_blas_f64(rows, columns, A, lda, S, U, ldu, VT, ldv);
#else
    detail::svd_scalar_f64(rows, columns, A, lda, S, U, ldu, VT, ldv);
#endif
}
}  // namespace linalg
