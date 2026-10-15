#include "include/matrix_operation/pseudo_inverse.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "include/common/workspace_arena.h"
#include "include/matrix_operation/svd_decomposition.h"
#include <include/logging.h>

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

template <typename T>
size_t pseudo_inverse_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto svd_bytes =
        linalg::svd_backend<T, linalg::kActiveBackend>::workspace_size(rows, columns);
    detail::workspace_arena arena;
    pseudo_inverse_plan<T>(arena, rows, columns, svd_bytes);
    return arena.total_size();
}

template <typename T>
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
        linalg::svd_backend<T, linalg::kActiveBackend>::workspace_size(rows, columns);

    LOGGING_CHECK(workspace_size >= pseudo_inverse_workspace_size_impl<T>(rows, columns),
        "pseudo_inverse: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = pseudo_inverse_plan<T>(arena, rows, columns, svd_bytes);

    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            bufs.A_copy[i * columns + j] = A[i * lda + j];
        }
    }

    linalg::svd_backend<T, linalg::kActiveBackend>::apply(rows,
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

    for (linalg_long c = 0; c < columns; ++c)
    {
        for (linalg_long r = 0; r < rows; ++r)
        {
            T sum = T(0);
            for (linalg_long i = 0; i < k; ++i)
            {
                const T s = bufs.S[i];
                if (s > eff_tol)
                {
                    sum += bufs.VT[i * columns + c] * (T(1) / s) * bufs.U[r * k + i];
                }
            }
            Ainv[c * ldai + r] = sum;
        }
    }
}

template <>
void pseudo_inverse_backend<float, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                               columns,
    const float*                                                              A,
    linalg_long                                                               lda,
    float*                                                                    Ainv,
    linalg_long                                                               ldai,
    float                                                                     tol,
    void*                                                                     workspace,
    size_t                                                                    workspace_size)
{
    LOGGING_CHECK(A != nullptr && Ainv != nullptr,
        "pseudo_inverse_backend::apply: A and Ainv must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0,
        "pseudo_inverse_backend::apply: rows/columns/lda/ldai must be positive");
    pseudo_inverse_impl(rows, columns, A, lda, Ainv, ldai, tol, workspace, workspace_size);
}

template <>
void pseudo_inverse_backend<double, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                                columns,
    const double*                                                              A,
    linalg_long                                                                lda,
    double*                                                                    Ainv,
    linalg_long                                                                ldai,
    double                                                                     tol,
    void*                                                                      workspace,
    size_t                                                                     workspace_size)
{
    LOGGING_CHECK(A != nullptr && Ainv != nullptr,
        "pseudo_inverse_backend::apply: A and Ainv must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldai != 0,
        "pseudo_inverse_backend::apply: rows/columns/lda/ldai must be positive");
    pseudo_inverse_impl(rows, columns, A, lda, Ainv, ldai, tol, workspace, workspace_size);
}

template <>
size_t pseudo_inverse_backend<float, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<float>(rows, columns);
}

template <>
size_t pseudo_inverse_backend<double, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg
