#include "include/matrix_operation/pseudo_inverse.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "include/common/workspace_arena.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "include/common/linalg_checks.h"
#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

#include "include/common/linalg_profiler.h"

namespace linalg
{

template <typename T> struct pseudo_inverse_layout
{
    T*             A_copy;
    T*             U;
    T*             VT;
    T*             S;
    unsigned char* svd_workspace;
};

template <typename T>
pseudo_inverse_layout<T> pseudo_inverse_plan(detail::workspace_arena& arena,
    linalg_long                                                       rows,
    linalg_long                                                       columns,
    size_t                                                            svd_workspace_bytes)
{
    const auto               k = std::min(rows, columns);
    pseudo_inverse_layout<T> p;
    p.A_copy        = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.U             = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
    p.VT            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
    p.S             = arena.next<T>(static_cast<size_t>(k));
    p.svd_workspace = arena.next<unsigned char>(svd_workspace_bytes);
    return p;
}

template <typename T, linalg_backend Backend>
size_t pseudo_inverse_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto svd_bytes =
        linalg::svd_decomposition<T, Backend>::workspace_size(rows, columns);
    detail::workspace_arena arena;
    pseudo_inverse_plan<T>(arena, rows, columns, svd_bytes);
    return arena.total_size();
}

template <typename T, linalg_backend Backend>
void pseudo_inverse_impl(linalg_long rows,
    linalg_long                      columns,
    const T*                         A,
    linalg_long                      lda,
    T*                               Ainv,
    linalg_long                      ldai,
    T                                tol,
    void*                            workspace,
    size_t                           workspace_size)
{
    const linalg_long k = std::min(rows, columns);
    const auto        svd_bytes =
        linalg::svd_decomposition<T, Backend>::workspace_size(rows, columns);

    const size_t needed = pseudo_inverse_workspace_size_impl<T, Backend>(rows, columns);
    LINALG_CHECK(workspace_size >= needed, "pseudo_inverse: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = pseudo_inverse_plan<T>(arena, rows, columns, svd_bytes);

    for (linalg_long i = 0; i < rows; ++i)
    {
        std::memcpy(bufs.A_copy + i * columns, A + i * lda,
                    static_cast<size_t>(columns) * sizeof(T));
    }

    linalg::svd_decomposition<T, Backend>::apply(rows,
        columns,
        bufs.A_copy,
        columns,
        bufs.S,
        bufs.U,
        k,
        bufs.VT,
        columns,
        bufs.svd_workspace,
        svd_bytes);

    T smax = T(0);
    for (linalg_long i = 0; i < k; ++i)
    {
        smax = std::max(smax, std::fabs(bufs.S[i]));
    }

    const T eff_tol = (tol >= T(0)) ? tol
                                    : std::numeric_limits<T>::epsilon() *
                                          static_cast<T>(std::max(rows, columns)) * smax;

    // Pre-scale VT rows by 1/S[i], storing into W (A_copy reused; SVD is done with it).
    // Absorbs both the per-element division and the threshold check into an O(k*columns)
    // pass, so the reconstruction loop below needs no division or branch.
    T* const W = bufs.A_copy;
#if LINALG_HAS_VECTORIZATION
    {
        using P                       = packet<T>;
        constexpr linalg_long vpw     = static_cast<linalg_long>(P::size);
        typename P::packet_t  zero_v;
        P::set(T(0), zero_v);
        for (linalg_long i = 0; i < k; ++i)
        {
            const T              inv_s  = (bufs.S[i] > eff_tol) ? T(1) / bufs.S[i] : T(0);
            const T*             VT_row = bufs.VT + i * columns;
            T*                   W_row  = W + i * columns;
            typename P::packet_t inv_sv;
            P::set(inv_s, inv_sv);
            linalg_long c = 0;
            for (; c + vpw <= columns; c += vpw)
            {
                typename P::packet_t vt, out;
                P::loadu(VT_row + c, vt);
                P::fma(inv_sv, vt, zero_v, out);
                P::storeu(out, W_row + c);
            }
            for (; c < columns; ++c)
                W_row[c] = inv_s * VT_row[c];
        }
    }
#else
    for (linalg_long i = 0; i < k; ++i)
    {
        const T      inv_s  = (bufs.S[i] > eff_tol) ? T(1) / bufs.S[i] : T(0);
        const T*     VT_row = bufs.VT + i * columns;
        T*           W_row  = W + i * columns;
        for (linalg_long c = 0; c < columns; ++c)
            W_row[c] = inv_s * VT_row[c];
    }
#endif

    // Ainv (columns x rows, col-major) = W^T (columns x k) * U^T (k x rows).
    // U_row[i] is contiguous (loadu); W column c has stride `columns` (gather).
    for (linalg_long c = 0; c < columns; ++c)
    {
        T* const     Ainv_col = Ainv + c * ldai;
        const T*     W_col    = W + c;
#if LINALG_HAS_VECTORIZATION
        {
            using P                   = packet<T>;
            constexpr linalg_long vpw = static_cast<linalg_long>(P::size);
            const linalg_int      col_stride = static_cast<linalg_int>(columns);
            for (linalg_long r = 0; r < rows; ++r)
            {
                const T* const       U_row = bufs.U + r * k;
                typename P::packet_t acc;
                P::set(T(0), acc);
                linalg_long i = 0;
                for (; i + vpw <= k; i += vpw)
                {
                    typename P::packet_t u_v, w_v;
                    P::loadu(U_row + i, u_v);
                    P::gather(W_col + i * columns, col_stride, w_v);
                    P::fma(u_v, w_v, acc, acc);
                }
                T sum = P::predux(acc);
                for (; i < k; ++i)
                    sum += U_row[i] * W_col[i * columns];
                Ainv_col[r] = sum;
            }
        }
#else
        for (linalg_long r = 0; r < rows; ++r)
        {
            const T* const U_row = bufs.U + r * k;
            T              sum   = T(0);
            for (linalg_long i = 0; i < k; ++i)
                sum += W[i * columns + c] * U_row[i];
            Ainv_col[r] = sum;
        }
#endif
    }
}

template <>
void pseudo_inverse<float, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                               columns,
    const float*                                                              A,
    linalg_long                                                               lda,
    float*                                                                    Ainv,
    linalg_long                                                               ldai,
    float                                                                     tol,
    void*                                                                     workspace,
    size_t                                                                    workspace_size)
{
    LINALG_SCOPE("pseudo_inverse/native<float>");
    LINALG_CHECK(A != nullptr && Ainv != nullptr,
        "pseudo_inverse::apply: A and Ainv must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0,
        "pseudo_inverse::apply: rows/columns/lda/ldai must be positive");
    pseudo_inverse_impl<float, linalg_backend::native>(
        rows, columns, A, lda, Ainv, ldai, tol, workspace, workspace_size);
}

template <>
void pseudo_inverse<double, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                                columns,
    const double*                                                              A,
    linalg_long                                                                lda,
    double*                                                                    Ainv,
    linalg_long                                                                ldai,
    double                                                                     tol,
    void*                                                                      workspace,
    size_t                                                                     workspace_size)
{
    LINALG_SCOPE("pseudo_inverse/native<double>");
    LINALG_CHECK(A != nullptr && Ainv != nullptr,
        "pseudo_inverse::apply: A and Ainv must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0,
        "pseudo_inverse::apply: rows/columns/lda/ldai must be positive");
    pseudo_inverse_impl<double, linalg_backend::native>(
        rows, columns, A, lda, Ainv, ldai, tol, workspace, workspace_size);
}

template <>
size_t pseudo_inverse<float, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<float, linalg_backend::native>(rows, columns);
}

template <>
size_t pseudo_inverse<double, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<double, linalg_backend::native>(rows, columns);
}

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

template <>
void pseudo_inverse<float, linalg_backend::blas_lapacke>::apply(linalg_long rows,
    linalg_long                                                                        columns,
    const float*                                                                       A,
    linalg_long                                                                        lda,
    float*                                                                             Ainv,
    linalg_long                                                                        ldai,
    float                                                                              tol,
    void*                                                                              workspace,
    size_t                                                                             workspace_size)
{
    LINALG_SCOPE("pseudo_inverse/blas_lapacke<float>");
    LINALG_CHECK(A != nullptr && Ainv != nullptr,
        "pseudo_inverse::apply: A and Ainv must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0,
        "pseudo_inverse::apply: rows/columns/lda/ldai must be positive");
    pseudo_inverse_impl<float, linalg_backend::blas_lapacke>(
        rows, columns, A, lda, Ainv, ldai, tol, workspace, workspace_size);
}

template <>
void pseudo_inverse<double, linalg_backend::blas_lapacke>::apply(linalg_long rows,
    linalg_long                                                                         columns,
    const double*                                                                       A,
    linalg_long                                                                         lda,
    double*                                                                             Ainv,
    linalg_long                                                                         ldai,
    double                                                                              tol,
    void*                                                                               workspace,
    size_t                                                                              workspace_size)
{
    LINALG_SCOPE("pseudo_inverse/blas_lapacke<double>");
    LINALG_CHECK(A != nullptr && Ainv != nullptr,
        "pseudo_inverse::apply: A and Ainv must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0,
        "pseudo_inverse::apply: rows/columns/lda/ldai must be positive");
    pseudo_inverse_impl<double, linalg_backend::blas_lapacke>(
        rows, columns, A, lda, Ainv, ldai, tol, workspace, workspace_size);
}

template <>
size_t pseudo_inverse<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<float, linalg_backend::blas_lapacke>(rows, columns);
}

template <>
size_t pseudo_inverse<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<double, linalg_backend::blas_lapacke>(rows, columns);
}

#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#if LINALG_HAS_MKL

template <>
void pseudo_inverse<float, linalg_backend::mkl>::apply(linalg_long rows,
    linalg_long                                                               columns,
    const float*                                                              A,
    linalg_long                                                               lda,
    float*                                                                    Ainv,
    linalg_long                                                               ldai,
    float                                                                     tol,
    void*                                                                     workspace,
    size_t                                                                    workspace_size)
{
    LINALG_SCOPE("pseudo_inverse/mkl<float>");
    LINALG_CHECK(A != nullptr && Ainv != nullptr,
        "pseudo_inverse::apply: A and Ainv must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0,
        "pseudo_inverse::apply: rows/columns/lda/ldai must be positive");
    pseudo_inverse_impl<float, linalg_backend::mkl>(
        rows, columns, A, lda, Ainv, ldai, tol, workspace, workspace_size);
}

template <>
void pseudo_inverse<double, linalg_backend::mkl>::apply(linalg_long rows,
    linalg_long                                                                columns,
    const double*                                                              A,
    linalg_long                                                                lda,
    double*                                                                    Ainv,
    linalg_long                                                                ldai,
    double                                                                     tol,
    void*                                                                      workspace,
    size_t                                                                     workspace_size)
{
    LINALG_SCOPE("pseudo_inverse/mkl<double>");
    LINALG_CHECK(A != nullptr && Ainv != nullptr,
        "pseudo_inverse::apply: A and Ainv must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0,
        "pseudo_inverse::apply: rows/columns/lda/ldai must be positive");
    pseudo_inverse_impl<double, linalg_backend::mkl>(
        rows, columns, A, lda, Ainv, ldai, tol, workspace, workspace_size);
}

template <>
size_t pseudo_inverse<float, linalg_backend::mkl>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<float, linalg_backend::mkl>(rows, columns);
}

template <>
size_t pseudo_inverse<double, linalg_backend::mkl>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<double, linalg_backend::mkl>(rows, columns);
}

#endif  // LINALG_HAS_MKL

}  // namespace linalg
