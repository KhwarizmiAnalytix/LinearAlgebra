#include "include/matrix_operation/matrix_rank.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "include/common/workspace_arena.h"
#include "include/matrix_operation/svd_decomposition.h"
#include <include/logging.h>

namespace linalg
{
namespace detail
{

template <typename T>
struct singular_values_layout
{
    T*             A_copy;
    T*             U;
    T*             VT;
    T*             S;
    unsigned char* svd_workspace;
};

template <typename T>
singular_values_layout<T> singular_values_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns, size_t svd_workspace_bytes)
{
    const auto k = std::min(rows, columns);
    singular_values_layout<T> p;
    p.A_copy        = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    p.U             = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
    p.VT            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
    p.S             = arena.next<T>(static_cast<size_t>(k));
    p.svd_workspace = arena.next<unsigned char>(svd_workspace_bytes);
    return p;
}

template <typename T>
size_t singular_values_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto svd_bytes =
        svd_decomposition_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    detail::workspace_arena arena;
    singular_values_plan<T>(arena, rows, columns, svd_bytes);
    return arena.total_size();
}

template <typename T> T* singular_values(const T* A, linalg_long rows, linalg_long columns,
    linalg_long lda, void* workspace, size_t workspace_size)
{
    const linalg_long k = std::min(rows, columns);
    const auto svd_bytes = svd_decomposition_workspace_size(static_cast<const T*>(nullptr), rows, columns);

    LOGGING_CHECK(workspace_size >= singular_values_workspace_size_impl<T>(rows, columns),
        "matrix_rank/matrix_condition_number: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = singular_values_plan<T>(arena, rows, columns, svd_bytes);

    for (linalg_long i = 0; i < rows; ++i)
    {
        for (linalg_long j = 0; j < columns; ++j)
        {
            bufs.A_copy[i * columns + j] = A[i * lda + j];
        }
    }

    linalg::svd_decomposition(rows, columns, bufs.A_copy, columns, bufs.S, bufs.U, k, bufs.VT,
        columns, bufs.svd_workspace, svd_bytes);
    return bufs.S;
}

template <typename T> linalg_long matrix_rank_impl(const T* A, linalg_long rows, linalg_long columns,
    linalg_long lda, T tol, void* workspace, size_t workspace_size)
{
    const linalg_long k = std::min(rows, columns);
    const T*          S = singular_values(A, rows, columns, lda, workspace, workspace_size);

    T smax = T(0);
    for (linalg_long i = 0; i < k; ++i)
    {
        smax = std::max(smax, std::fabs(S[i]));
    }

    const T eff_tol =
        (tol >= T(0)) ? tol : std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(rows, columns)) * smax;

    linalg_long rank = 0;
    for (linalg_long i = 0; i < k; ++i)
    {
        if (std::fabs(S[i]) > eff_tol)
        {
            ++rank;
        }
    }
    return rank;
}

template <typename T> T matrix_condition_number_impl(const T* A, linalg_long rows, linalg_long columns,
    linalg_long lda, void* workspace, size_t workspace_size)
{
    const linalg_long k = std::min(rows, columns);
    const T*          S = singular_values(A, rows, columns, lda, workspace, workspace_size);

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

}  // namespace detail

size_t matrix_rank_workspace_size(const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return detail::singular_values_workspace_size_impl<float>(rows, columns);
}

size_t matrix_rank_workspace_size(const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return detail::singular_values_workspace_size_impl<double>(rows, columns);
}

//-----------------------------------------------------------------------------
linalg_long matrix_rank(const float* A, linalg_long rows, linalg_long columns, linalg_long lda,
    float tol, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(A != nullptr, "matrix_rank: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0, "matrix_rank: rows/columns/lda must be positive");
    return detail::matrix_rank_impl(A, rows, columns, lda, tol, workspace, workspace_size);
}

//-----------------------------------------------------------------------------
linalg_long matrix_rank(const double* A, linalg_long rows, linalg_long columns, linalg_long lda,
    double tol, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(A != nullptr, "matrix_rank: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0, "matrix_rank: rows/columns/lda must be positive");
    return detail::matrix_rank_impl(A, rows, columns, lda, tol, workspace, workspace_size);
}

//-----------------------------------------------------------------------------
float matrix_condition_number(const float* A, linalg_long rows, linalg_long columns, linalg_long lda,
    void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(A != nullptr, "matrix_condition_number: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0, "matrix_condition_number: rows/columns/lda must be positive");
    return detail::matrix_condition_number_impl(A, rows, columns, lda, workspace, workspace_size);
}

//-----------------------------------------------------------------------------
double matrix_condition_number(const double* A, linalg_long rows, linalg_long columns, linalg_long lda,
    void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(A != nullptr, "matrix_condition_number: A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0, "matrix_condition_number: rows/columns/lda must be positive");
    return detail::matrix_condition_number_impl(A, rows, columns, lda, workspace, workspace_size);
}

}  // namespace linalg
