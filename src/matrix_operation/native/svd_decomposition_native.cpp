#include "include/matrix_operation/svd_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/workspace_arena.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace linalg
{

// Null/positivity checks shared by every svd_decomposition<T, Backend>::apply
// below, so callers (tests included) get the same validation regardless of
// which backend is active. There is no non-detail svd_decomposition()
// wrapper to put this in any more — callers use
// svd_decomposition<T, Backend>::apply directly, naming Backend explicitly.
static void check_svd_inputs(const void* A,
    const void*                          S,
    const void*                          U,
    const void*                          VT,
    linalg_long                          rows,
    linalg_long                          columns,
    linalg_long                          lda,
    linalg_long                          ldu,
    linalg_long                          ldv)
{
    LINALG_CHECK(A != nullptr && S != nullptr && U != nullptr && VT != nullptr,
        "svd_decomposition::apply: A, S, U, and VT must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldu != 0 && ldv != 0,
        "svd_decomposition::apply: rows/columns/lda/ldu/ldv must be positive");
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/svd_decomposition_mkl.cpp and
// blas/svd_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

#define U(i, j) U_[(i) * dim[0] + (j)]
#define S(i, j) S_[(i) * dim[1] + (j)]
#define V(i, j) V_[(i) * dim[1] + (j)]

template <class T> void GivensL(T* S_, const linalg_long dim[2], linalg_long m, T a, T b)  // NOLINT
{
    T r = sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

#pragma omp parallel for
    for (linalg_int i = 0; i < static_cast<linalg_int>(dim[1]); i++)
    {
        T S0 = S(m + 0, i);
        T S1 = S(m + 1, i);
        S(m, i) += S0 * (c - 1);
        S(m, i) += S1 * (-s);

        S(m + 1, i) += S0 * (s);
        S(m + 1, i) += S1 * (c - 1);
    }
}

template <class T> void GivensR(T* S_, const linalg_long dim[2], linalg_long m, T a, T b)  // NOLINT
{
    T r = sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

#pragma omp parallel for
    for (linalg_int i = 0; i < static_cast<linalg_int>(dim[0]); i++)
    {
        T S0 = S(i, m + 0);
        T S1 = S(i, m + 1);
        S(i, m) += S0 * (c - 1);
        S(i, m) += S1 * (-s);

        S(i, m + 1) += S0 * (s);
        S(i, m + 1) += S1 * (c - 1);
    }
}

// `house_vec` (max(dim[0],dim[1]) elements) is caller-owned scratch.
template <class T>
void SVD(const linalg_long dim[2], T* U_, T* S_, T* V_, T* house_vec, T eps = -1)  // NOLINT
{
    assert(dim[0] >= dim[1]);

    {  // Bi-diagonalization
        const auto n = std::min(dim[0], dim[1]);
        for (linalg_long i = 0; i < n; i++)
        {
            // Column Householder
            {
                T x1 = S(i, i);
                if (x1 < 0)
                {
                    x1 = -x1;
                }

                T x_inv_norm = 0;
                for (linalg_long j = i; j < dim[0]; j++)
                {
                    x_inv_norm += S(j, i) * S(j, i);
                }
                if (x_inv_norm > 0)
                {
                    // Non-degenerate subcolumn: build the unit Householder
                    // vector as usual.
                    x_inv_norm = 1 / sqrt(x_inv_norm);

                    T alpha = sqrt(1 + x1 * x_inv_norm);
                    T beta  = x_inv_norm / alpha;

                    house_vec[i] = -alpha;
                    for (linalg_long j = i + 1; j < dim[0]; j++)
                    {
                        house_vec[j] = -beta * S(j, i);
                    }
                    if (S(i, i) < 0)
                    {
                        for (linalg_long j = i + 1; j < dim[0]; j++)
                        {
                            house_vec[j] = -house_vec[j];
                        }
                    }
                }
                else
                {
                    // Subcolumn (including the pivot) is already exactly
                    // zero: there is nothing to reflect. Falling through to
                    // alpha = sqrt(1 + 0) = 1 here would fabricate a
                    // spurious non-identity reflection (house_vec[i] = -1)
                    // that corrupts an already-zero entry instead of
                    // leaving it alone — this hit e.g. the identity matrix,
                    // where the bidiagonalization has nothing left to do.
                    for (linalg_long j = i; j < dim[0]; j++)
                    {
                        house_vec[j] = 0;
                    }
                }
            }
#pragma omp parallel for
            for (auto k = static_cast<linalg_int>(i); k < static_cast<linalg_int>(dim[1]); k++)
            {
                T dot_prod = 0;
                for (linalg_long j = i; j < dim[0]; j++)
                {
                    dot_prod += S(j, k) * house_vec[j];
                }
                for (linalg_long j = i; j < dim[0]; j++)
                {
                    S(j, k) -= dot_prod * house_vec[j];
                }
            }
#pragma omp parallel for
            for (linalg_int k = 0; k < static_cast<linalg_int>(dim[0]); k++)
            {
                T dot_prod = 0;
                for (linalg_long j = i; j < dim[0]; j++)
                {
                    dot_prod += U(k, j) * house_vec[j];
                }
                for (linalg_long j = i; j < dim[0]; j++)
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
                for (linalg_long j = i + 1; j < dim[1]; j++)
                {
                    x_inv_norm += S(i, j) * S(i, j);
                }
                if (x_inv_norm > 0)
                {
                    // Non-degenerate subrow: build the unit Householder
                    // vector as usual.
                    x_inv_norm = 1 / sqrt(x_inv_norm);

                    T alpha = sqrt(1 + x1 * x_inv_norm);
                    T beta  = x_inv_norm / alpha;

                    house_vec[i + 1] = -alpha;
                    for (linalg_long j = i + 2; j < dim[1]; j++)
                    {
                        house_vec[j] = -beta * S(i, j);
                    }
                    if (S(i, i + 1) < 0)
                    {
                        for (linalg_long j = i + 2; j < dim[1]; j++)
                        {
                            house_vec[j] = -house_vec[j];
                        }
                    }
                }
                else
                {
                    // Subrow (including the pivot) is already exactly
                    // zero — see the matching column-Householder branch
                    // above for why this must be a no-op rather than
                    // falling through to a spurious alpha = 1 reflection.
                    for (linalg_long j = i + 1; j < dim[1]; j++)
                    {
                        house_vec[j] = 0;
                    }
                }
            }
#pragma omp parallel for
            for (auto k = static_cast<linalg_int>(i); k < static_cast<linalg_int>(dim[0]); k++)
            {
                T dot_prod = 0;
                for (linalg_long j = i + 1; j < dim[1]; j++)
                {
                    dot_prod += S(k, j) * house_vec[j];
                }
                for (linalg_long j = i + 1; j < dim[1]; j++)
                {
                    S(k, j) -= dot_prod * house_vec[j];
                }
            }
#pragma omp parallel for
            for (linalg_int k = 0; k < static_cast<linalg_int>(dim[1]); k++)
            {
                T dot_prod = 0;
                for (linalg_long j = i + 1; j < dim[1]; j++)
                {
                    dot_prod += V(j, k) * house_vec[j];
                }
                for (linalg_long j = i + 1; j < dim[1]; j++)
                {
                    V(j, k) -= dot_prod * house_vec[j];
                }
            }
        }
    }

    linalg_long k0 = 0;
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
        for (linalg_long i = 0; i < dim[1]; i++)
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

        linalg_long n = k0 + 2;
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

        for (linalg_long k = k0; k < n - 1; k++)
        {
            linalg_long dimU[2] = {dim[0], dim[0]};  // NOLINT
            linalg_long dimV[2] = {dim[1], dim[1]};  // NOLINT
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
            for (linalg_long i0 = k0; i0 < n - 1; i0++)
            {
                for (linalg_long i1 = 0; i1 < dim[1]; i1++)
                {
                    if (i0 > i1 || i0 + 1 < i1)
                    {
                        S(i0, i1) = 0;
                    }
                }
            }
            for (linalg_long i0 = 0; i0 < dim[0]; i0++)
            {
                for (linalg_long i1 = k0; i1 < n - 1; i1++)
                {
                    if (i0 > i1 || i0 + 1 < i1)
                    {
                        S(i0, i1) = 0;
                    }
                }
            }
            for (linalg_long i = 0; i < dim[1] - 1; i++)
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

template <typename T> struct svd_scalar_layout
{
    T* U_;
    T* V_;
    T* S_;
    T* house_vec;
};

template <typename T>
svd_scalar_layout<T> svd_scalar_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns)
{
    const linalg_long    dim[2] = {std::max(rows, columns), std::min(rows, columns)};  // NOLINT
    svd_scalar_layout<T> p;
    p.U_        = arena.next<T>(static_cast<size_t>(dim[0]) * static_cast<size_t>(dim[0]));
    p.V_        = arena.next<T>(static_cast<size_t>(dim[1]) * static_cast<size_t>(dim[1]));
    p.S_        = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.house_vec = arena.next<T>(static_cast<size_t>(std::max(dim[0], dim[1])));
    return p;
}

template <typename T> size_t svd_scalar_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    svd_scalar_plan<T>(arena, rows, columns);
    return arena.total_size();
}

template <class T>
inline void svd(  // NOLINT
    linalg_long rows,
    linalg_long columns,
    T*          A,
    linalg_long lda,
    T*          S,
    T*          U,
    linalg_long ldu,
    T*          VT,
    linalg_long ldv,
    void*       workspace,
    size_t      workspace_size)
{
    const linalg_long dim[2] = {std::max(rows, columns), std::min(rows, columns)};  // NOLINT

    LINALG_CHECK(workspace_size >= svd_scalar_workspace_size_impl<T>(rows, columns),
        "svd_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = svd_scalar_plan<T>(arena, rows, columns);
    auto* const             U_   = bufs.U_;
    auto* const             V_   = bufs.V_;
    auto* const             S_   = bufs.S_;

    memset(U_, 0, dim[0] * dim[0] * sizeof(T));
    memset(V_, 0, dim[1] * dim[1] * sizeof(T));

    if (dim[1] == columns)
    {
        for (linalg_long i = 0; i < dim[0]; i++)
        {
            for (linalg_long j = 0; j < dim[1]; j++)
            {
                S_[i * dim[1] + j] = A[i * lda + j];
            }
        }
    }
    else
    {
        for (linalg_long i = 0; i < dim[0]; i++)
        {
            for (linalg_long j = 0; j < dim[1]; j++)
            {
                S_[i * dim[1] + j] = A[j * lda + i];
            }
        }
    }

    for (linalg_long i = 0; i < dim[0]; i++)
    {
        U_[i * dim[0] + i] = 1;
    }
    for (linalg_long i = 0; i < dim[1]; i++)
    {
        V_[i * dim[1] + i] = 1;
    }

    SVD<T>(dim, U_, S_, V_, bufs.house_vec, (T)-1);

    for (linalg_long i = 0; i < dim[1]; i++)
    {  // Set S
        S[i] = S_[i * dim[1] + i];
    }
    if (dim[1] == columns)
    {  // A was factored directly: U_ holds left vectors, V_ holds VT.
        for (linalg_long i = 0; i < rows; i++)
        {
            for (linalg_long j = 0; j < columns; j++)
            {
                U[j + ldu * i] = U_[j + i * rows];
            }
        }
        for (linalg_long i = 0; i < columns; i++)
        {
            for (linalg_long j = 0; j < columns; j++)
            {
                VT[j + ldv * i] = V_[j + i * columns] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
            }
        }
    }
    else
    {  // A was transposed before the factorization, so the factors swap roles.
        for (linalg_long i = 0; i < rows; i++)
        {
            for (linalg_long j = 0; j < rows; j++)
            {
                U[j + ldu * i] = V_[i + j * rows];
            }
        }
        for (linalg_long i = 0; i < rows; i++)
        {
            for (linalg_long j = 0; j < columns; j++)
            {
                VT[j + ldv * i] = U_[i + j * columns] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
            }
        }
    }

    for (linalg_long i = 0; i < dim[1]; i++)
    {
        S[i] = S[i] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
    }
}

}  // namespace

template <>
void svd_decomposition<float, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                          columns,
    float*                                                               A,
    linalg_long                                                          lda,
    float*                                                               S,
    float*                                                               U,
    linalg_long                                                          ldu,
    float*                                                               VT,
    linalg_long                                                          ldv,
    void*                                                                workspace,
    size_t                                                               workspace_size)
{
    check_svd_inputs(A, S, U, VT, rows, columns, lda, ldu, ldv);
    svd(rows, columns, A, lda, S, U, ldu, VT, ldv, workspace, workspace_size);
}

template <>
void svd_decomposition<double, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                           columns,
    double*                                                               A,
    linalg_long                                                           lda,
    double*                                                               S,
    double*                                                               U,
    linalg_long                                                           ldu,
    double*                                                               VT,
    linalg_long                                                           ldv,
    void*                                                                 workspace,
    size_t                                                                workspace_size)
{
    check_svd_inputs(A, S, U, VT, rows, columns, lda, ldu, ldv);
    svd(rows, columns, A, lda, S, U, ldu, VT, ldv, workspace, workspace_size);
}

template <>
size_t svd_decomposition<float, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return svd_scalar_workspace_size_impl<float>(rows, columns);
}

template <>
size_t svd_decomposition<double, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return svd_scalar_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg
