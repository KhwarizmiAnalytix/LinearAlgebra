#include "include/matrix_operation/matrix_norm.h"

#include <algorithm>
#include <cmath>

#include "include/common/workspace_arena.h"
#include "include/matrix_operation/svd_decomposition.h"
#include <include/logging.h>

namespace linalg
{

template <typename T>
T norm_frobenius(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    T sum = T(0);
    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            const T v = A[i * lda + j];
            sum += v * v;
        }
    }
    return std::sqrt(sum);
}

template <typename T> T norm_one(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    T best = T(0);
    for (linalg_long j = 0; j < columns; ++j)
    {
        T col_sum = T(0);
        for (linalg_long i = 0; i < rows; ++i)
        {
            col_sum += std::fabs(A[i * lda + j]);
        }
        best = std::max(best, col_sum);
    }
    return best;
}

template <typename T>
T norm_infinity(const T* A, linalg_long rows, linalg_long columns, linalg_long lda)
{
    T best = T(0);
    for (linalg_long i = 0; i < rows; ++i)
    {
        T row_sum = T(0);
        for (linalg_long j = 0; j < columns; ++j)
        {
            row_sum += std::fabs(A[i * lda + j]);
        }
        best = std::max(best, row_sum);
    }
    return best;
}

template <typename T> struct norm_two_layout
{
    T*             A_copy;
    T*             U;
    T*             VT;
    T*             S;
    unsigned char* svd_workspace;
};

template <typename T>
norm_two_layout<T> norm_two_plan(detail::workspace_arena& arena,
    linalg_long                                           rows,
    linalg_long                                           columns,
    size_t                                                svd_workspace_bytes)
{
    const auto         k = std::min(rows, columns);
    norm_two_layout<T> p;
    p.A_copy        = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.U             = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
    p.VT            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
    p.S             = arena.next<T>(static_cast<size_t>(k));
    p.svd_workspace = arena.next<unsigned char>(svd_workspace_bytes);
    return p;
}

template <typename T> size_t norm_two_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto svd_bytes =
        linalg::svd_backend<T, linalg::kActiveBackend>::workspace_size(rows, columns);
    detail::workspace_arena arena;
    norm_two_plan<T>(arena, rows, columns, svd_bytes);
    return arena.total_size();
}

template <typename T>
T norm_two(const T* A,
    linalg_long     rows,
    linalg_long     columns,
    linalg_long     lda,
    void*           workspace,
    size_t          workspace_size)
{
    const linalg_long k = std::min(rows, columns);
    const auto        svd_bytes =
        linalg::svd_backend<T, linalg::kActiveBackend>::workspace_size(rows, columns);

    LOGGING_CHECK(workspace_size >= norm_two_workspace_size_impl<T>(rows, columns),
        "matrix_norm: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = norm_two_plan<T>(arena, rows, columns, svd_bytes);

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

    T best = T(0);
    for (linalg_long i = 0; i < k; ++i)
    {
        best = std::max(best, std::fabs(bufs.S[i]));
    }
    return best;
}

template <typename T>
size_t matrix_norm_workspace_size_impl(linalg_long rows, linalg_long columns, matrix_norm_type type)
{
    switch (type)
    {
    case matrix_norm_type::TWO:
        return norm_two_workspace_size_impl<T>(rows, columns);
    default:
        return 0;
    }
}

template <typename T>
T matrix_norm_impl(const T* A,
    linalg_long             rows,
    linalg_long             columns,
    linalg_long             lda,
    matrix_norm_type        type,
    void*                   workspace,
    size_t                  workspace_size)
{
    switch (type)
    {
    case matrix_norm_type::FROBENIUS:
        return norm_frobenius(A, rows, columns, lda);
    case matrix_norm_type::ONE:
        return norm_one(A, rows, columns, lda);
    case matrix_norm_type::INFINITY_NORM:
        return norm_infinity(A, rows, columns, lda);
    case matrix_norm_type::TWO:
        return norm_two(A, rows, columns, lda, workspace, workspace_size);
    default:
        LOGGING_THROW("unsupported matrix_norm_type", static_cast<linalg_int>(type));
    }
}

template <>
float matrix_norm_backend<float, linalg_backend::native>::apply(const float* A,
    linalg_long                                                              rows,
    linalg_long                                                              columns,
    linalg_long                                                              lda,
    matrix_norm_type                                                         type,
    void*                                                                    workspace,
    size_t                                                                   workspace_size)
{
    LOGGING_CHECK(A != nullptr, "matrix_norm_backend::apply: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0,
        "matrix_norm_backend::apply: rows/columns/lda must be positive");
    return matrix_norm_impl(A, rows, columns, lda, type, workspace, workspace_size);
}

template <>
double matrix_norm_backend<double, linalg_backend::native>::apply(const double* A,
    linalg_long                                                                 rows,
    linalg_long                                                                 columns,
    linalg_long                                                                 lda,
    matrix_norm_type                                                            type,
    void*                                                                       workspace,
    size_t                                                                      workspace_size)
{
    LOGGING_CHECK(A != nullptr, "matrix_norm_backend::apply: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0,
        "matrix_norm_backend::apply: rows/columns/lda must be positive");
    return matrix_norm_impl(A, rows, columns, lda, type, workspace, workspace_size);
}

template <>
size_t matrix_norm_backend<float, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns, matrix_norm_type type)
{
    return matrix_norm_workspace_size_impl<float>(rows, columns, type);
}

template <>
size_t matrix_norm_backend<double, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns, matrix_norm_type type)
{
    return matrix_norm_workspace_size_impl<double>(rows, columns, type);
}

}  // namespace linalg
