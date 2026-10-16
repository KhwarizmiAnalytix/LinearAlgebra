#include "include/matrix_operation/qr_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/workspace_arena.h"

#include <algorithm>
#include <cmath>

namespace linalg
{

// Null/positivity checks shared by every qr_decomposition<T, Backend>::apply
// below, so callers (tests included) get the same validation regardless of
// which backend is active. There is no non-detail qr_decomposition()
// wrapper to put this in any more — callers use
// qr_decomposition<T, Backend>::apply directly, naming Backend explicitly.
static void check_qr_inputs(const void* A,
    const void*                         Q,
    const void*                         R,
    linalg_long                         rows,
    linalg_long                         columns,
    linalg_long                         lda,
    linalg_long                         ldq,
    linalg_long                         ldr)
{
    LINALG_CHECK(A != nullptr && Q != nullptr && R != nullptr,
        "qr_decomposition::apply: A, Q, and R must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldq != 0 && ldr != 0,
        "qr_decomposition::apply: rows/columns/lda/ldq/ldr must be positive");
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/qr_decomposition_mkl.cpp and
// blas/qr_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

template <typename T> struct qr_householder_layout
{
    T* Rwork;
    T* Qwork;
    T* v;
};

template <typename T>
qr_householder_layout<T> qr_householder_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns)
{
    qr_householder_layout<T> p;
    p.Rwork = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.Qwork = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(rows));
    p.v     = arena.next<T>(static_cast<size_t>(rows));
    return p;
}

template <typename T>
size_t qr_householder_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    qr_householder_plan<T>(arena, rows, columns);
    return arena.total_size();
}

// Householder QR: builds Q (rows x rows, accumulated as a product of
// reflectors) and an upper-triangular working copy of A, then the caller
// slices out the economy-size Q (first k columns) / R (first k rows).
template <typename T>
void qr_householder(linalg_long rows,
    linalg_long                 columns,
    const T*                    A,
    linalg_long                 lda,
    T*                          Q,
    linalg_long                 ldq,
    T*                          R,
    linalg_long                 ldr,
    void*                       workspace,
    size_t                      workspace_size)
{
    const linalg_long k = std::min(rows, columns);

    LINALG_CHECK(workspace_size >= qr_householder_workspace_size_impl<T>(rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs  = qr_householder_plan<T>(arena, rows, columns);
    auto* const             Rwork = bufs.Rwork;
    auto* const             Qwork = bufs.Qwork;
    auto* const             v     = bufs.v;

    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            Rwork[i * columns + j] = A[i * lda + j];
        }
    }

    std::fill_n(Qwork, rows * rows, T(0));
    for (linalg_long i = 0; i < rows; ++i)
    {
        Qwork[i * rows + i] = T(1);
    }

    for (linalg_long i = 0; i < k; ++i)
    {
        T norm = T(0);
        for (linalg_long r = i; r < rows; ++r)
        {
            const T x = Rwork[r * columns + i];
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm == T(0))
        {
            continue;
        }

        const T diag  = Rwork[i * columns + i];
        const T alpha = (diag >= T(0)) ? -norm : norm;

        for (linalg_long r = i; r < rows; ++r)
        {
            v[static_cast<std::size_t>(r)] = Rwork[r * columns + i];
        }
        v[static_cast<std::size_t>(i)] -= alpha;

        T vnorm = T(0);
        for (linalg_long r = i; r < rows; ++r)
        {
            vnorm += v[static_cast<std::size_t>(r)] * v[static_cast<std::size_t>(r)];
        }
        vnorm = std::sqrt(vnorm);
        if (vnorm == T(0))
        {
            continue;
        }
        for (linalg_long r = i; r < rows; ++r)
        {
            v[static_cast<std::size_t>(r)] /= vnorm;
        }

        // Apply H = I - 2*v*v^T to Rwork's columns [i, columns) over rows [i, rows).
        for (linalg_long j = i; j < columns; ++j)
        {
            T dot = T(0);
            for (linalg_long r = i; r < rows; ++r)
            {
                dot += v[static_cast<std::size_t>(r)] * Rwork[r * columns + j];
            }
            for (linalg_long r = i; r < rows; ++r)
            {
                Rwork[r * columns + j] -= T(2) * v[static_cast<std::size_t>(r)] * dot;
            }
        }

        // Accumulate Q = Q * H over Qwork's columns [i, rows).
        for (linalg_long r = 0; r < rows; ++r)
        {
            T dot = T(0);
            for (linalg_long c = i; c < rows; ++c)
            {
                dot += Qwork[r * rows + c] * v[static_cast<std::size_t>(c)];
            }
            for (linalg_long c = i; c < rows; ++c)
            {
                Qwork[r * rows + c] -= T(2) * dot * v[static_cast<std::size_t>(c)];
            }
        }
    }

    for (linalg_long r = 0; r < rows; ++r)
    {
        for (linalg_long c = 0; c < k; ++c)
        {
            Q[r * ldq + c] = Qwork[r * rows + c];
        }
    }

    for (linalg_long r = 0; r < k; ++r)
    {
        for (linalg_long c = 0; c < columns; ++c)
        {
            R[r * ldr + c] = (c >= r) ? Rwork[r * columns + c] : T(0);
        }
    }
}

}  // namespace

template <>
void qr_decomposition<float, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                   columns,
    const float*                                                  A,
    linalg_long                                                   lda,
    float*                                                        Q,
    linalg_long                                                   ldq,
    float*                                                        R,
    linalg_long                                                   ldr,
    void*                                                         workspace,
    size_t                                                        workspace_size)
{
    check_qr_inputs(A, Q, R, rows, columns, lda, ldq, ldr);
    qr_householder(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
}

template <>
void qr_decomposition<double, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                    columns,
    const double*                                                  A,
    linalg_long                                                    lda,
    double*                                                        Q,
    linalg_long                                                    ldq,
    double*                                                        R,
    linalg_long                                                    ldr,
    void*                                                          workspace,
    size_t                                                         workspace_size)
{
    check_qr_inputs(A, Q, R, rows, columns, lda, ldq, ldr);
    qr_householder(rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size);
}

template <>
size_t qr_decomposition<float, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return qr_householder_workspace_size_impl<float>(rows, columns);
}

template <>
size_t qr_decomposition<double, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return qr_householder_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg
