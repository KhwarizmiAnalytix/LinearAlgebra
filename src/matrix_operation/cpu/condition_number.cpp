#include "include/matrix_operation/condition_number.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "include/common/workspace_arena.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "include/common/linalg_checks.h"

#include "include/common/linalg_profiler.h"

namespace linalg
{

namespace
{

template <typename T> struct singular_values_layout
{
    T*             A_copy;
    T*             U;
    T*             VT;
    T*             S;
    unsigned char* svd_workspace;
};

template <typename T>
singular_values_layout<T> singular_values_plan(detail::workspace_arena& arena,
    linalg_long                                                         rows,
    linalg_long                                                         columns,
    size_t                                                              svd_workspace_bytes)
{
    const auto                k = std::min(rows, columns);
    singular_values_layout<T> p;
    p.A_copy        = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.U             = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
    p.VT            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
    p.S             = arena.next<T>(static_cast<size_t>(k));
    p.svd_workspace = arena.next<unsigned char>(svd_workspace_bytes);
    return p;
}

template <typename T, linalg_backend Backend>
size_t singular_values_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto svd_bytes =
        linalg::svd_decomposition<T, Backend>::workspace_size(rows, columns);
    detail::workspace_arena arena;
    singular_values_plan<T>(arena, rows, columns, svd_bytes);
    return arena.total_size();
}

template <typename T, linalg_backend Backend>
T* singular_values(const T* A,
    linalg_long             rows,
    linalg_long             columns,
    linalg_long             lda,
    void*                   workspace,
    size_t                  workspace_size)
{
    const linalg_long k = std::min(rows, columns);
    const auto        svd_bytes =
        linalg::svd_decomposition<T, Backend>::workspace_size(rows, columns);

    const size_t needed = singular_values_workspace_size_impl<T, Backend>(rows, columns);
    LINALG_CHECK(workspace_size >= needed, "condition_number: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = singular_values_plan<T>(arena, rows, columns, svd_bytes);

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
    return bufs.S;
}

template <typename T, linalg_backend Backend>
T matrix_condition_number_impl(const T* A,
    linalg_long                         rows,
    linalg_long                         columns,
    linalg_long                         lda,
    void*                               workspace,
    size_t                              workspace_size)
{
    const linalg_long k = std::min(rows, columns);
    const T* S = singular_values<T, Backend>(A, rows, columns, lda, workspace, workspace_size);

    T smax = T(0);
    T smin = std::numeric_limits<T>::max();
    for (linalg_long i = 0; i < k; ++i)
    {
        const T abs_s = std::fabs(S[i]);
        smax          = std::max(smax, abs_s);
        smin          = std::min(smin, abs_s);
    }

    if (smin == T(0))
    {
        return std::numeric_limits<T>::infinity();
    }
    return smax / smin;
}

}  // namespace

template <>
float condition_number<float, linalg_backend::NATIVE>::apply(const float* A,
    linalg_long                                                                          rows,
    linalg_long                                                                          columns,
    linalg_long                                                                          lda,
    void*                                                                                workspace,
    size_t workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("condition_number/native<float>");
    LINALG_CHECK(A != nullptr, "condition_number::apply: A must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0,
        "condition_number::apply: rows/columns/lda must be positive");
    return matrix_condition_number_impl<float, linalg_backend::NATIVE>(
        A, rows, columns, lda, workspace, workspace_size);
}

template <>
double condition_number<double, linalg_backend::NATIVE>::apply(const double* A,
    linalg_long                                                                             rows,
    linalg_long                                                                             columns,
    linalg_long                                                                             lda,
    void*  workspace,
    size_t workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("condition_number/native<double>");
    LINALG_CHECK(A != nullptr, "condition_number::apply: A must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0,
        "condition_number::apply: rows/columns/lda must be positive");
    return matrix_condition_number_impl<double, linalg_backend::NATIVE>(
        A, rows, columns, lda, workspace, workspace_size);
}

template <>
size_t condition_number<float, linalg_backend::NATIVE>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<float, linalg_backend::NATIVE>(rows, columns);
}

template <>
size_t condition_number<double, linalg_backend::NATIVE>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<double, linalg_backend::NATIVE>(rows, columns);
}

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

template <>
float condition_number<float, linalg_backend::BLAS_LAPACKE>::apply(const float* A,
    linalg_long                                                                              rows,
    linalg_long                                                                              columns,
    linalg_long                                                                              lda,
    void*                                                                                    workspace,
    size_t workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("condition_number/blas_lapacke<float>");
    LINALG_CHECK(A != nullptr, "condition_number::apply: A must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0,
        "condition_number::apply: rows/columns/lda must be positive");
    return matrix_condition_number_impl<float, linalg_backend::BLAS_LAPACKE>(
        A, rows, columns, lda, workspace, workspace_size);
}

template <>
double condition_number<double, linalg_backend::BLAS_LAPACKE>::apply(const double* A,
    linalg_long                                                                               rows,
    linalg_long                                                                               columns,
    linalg_long                                                                               lda,
    void*  workspace,
    size_t workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("condition_number/blas_lapacke<double>");
    LINALG_CHECK(A != nullptr, "condition_number::apply: A must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0,
        "condition_number::apply: rows/columns/lda must be positive");
    return matrix_condition_number_impl<double, linalg_backend::BLAS_LAPACKE>(
        A, rows, columns, lda, workspace, workspace_size);
}

template <>
size_t condition_number<float, linalg_backend::BLAS_LAPACKE>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<float, linalg_backend::BLAS_LAPACKE>(rows, columns);
}

template <>
size_t condition_number<double, linalg_backend::BLAS_LAPACKE>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<double, linalg_backend::BLAS_LAPACKE>(rows, columns);
}

#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#if LINALG_HAS_MKL

template <>
float condition_number<float, linalg_backend::MKL>::apply(const float* A,
    linalg_long                                                                   rows,
    linalg_long                                                                   columns,
    linalg_long                                                                   lda,
    void*                                                                         workspace,
    size_t                                                                        workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("condition_number/mkl<float>");
    LINALG_CHECK(A != nullptr, "condition_number::apply: A must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0,
        "condition_number::apply: rows/columns/lda must be positive");
    return matrix_condition_number_impl<float, linalg_backend::MKL>(
        A, rows, columns, lda, workspace, workspace_size);
}

template <>
double condition_number<double, linalg_backend::MKL>::apply(const double* A,
    linalg_long                                                                    rows,
    linalg_long                                                                    columns,
    linalg_long                                                                    lda,
    void*  workspace,
    size_t workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("condition_number/mkl<double>");
    LINALG_CHECK(A != nullptr, "condition_number::apply: A must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && lda != 0,
        "condition_number::apply: rows/columns/lda must be positive");
    return matrix_condition_number_impl<double, linalg_backend::MKL>(
        A, rows, columns, lda, workspace, workspace_size);
}

template <>
size_t condition_number<float, linalg_backend::MKL>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<float, linalg_backend::MKL>(rows, columns);
}

template <>
size_t condition_number<double, linalg_backend::MKL>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<double, linalg_backend::MKL>(rows, columns);
}

#endif  // LINALG_HAS_MKL

}  // namespace linalg
