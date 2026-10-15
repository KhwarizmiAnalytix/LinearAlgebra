
#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
#include "include/matrix_operation/svd_decomposition.h"

#include <include/logging.h>

#include "include/common/workspace_arena.h"

#include <algorithm>
#include <lapacke.h>

// BLAS/LAPACKE specializations for svd_decomposition.h (the native backend lives in
// svd_decomposition.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
// on; the #if below is kept as a second guard.
namespace linalg
{
// Same checks as svd_decomposition.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
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
    LOGGING_CHECK(A != nullptr && S != nullptr && U != nullptr && VT != nullptr,
        "svd_decomposition::apply: A, S, U, and VT must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && lda != 0 && ldu != 0 && ldv != 0,
        "svd_decomposition::apply: rows/columns/lda/ldu/ldv must be positive");
}

namespace
{

template <typename T> struct svd_layout
{
    T* work;
};

template <typename T> svd_layout<T> svd_plan(detail::workspace_arena& arena, lapack_int lwork)
{
    svd_layout<T> p;
    p.work = arena.next<T>(static_cast<size_t>(lwork));
    return p;
}

// gesvd's 'O'/'S' job choice (overwrite A in place with one factor) depends
// on rows > columns exactly as the real call below branches, since the
// optimal lwork LAPACK reports depends on which of U/VT is requested via
// jobu/jobvt.
template <typename T, typename GesvdWorkFn>
lapack_int svd_lwork(GesvdWorkFn gesvd_work, lapack_int r, lapack_int c, bool rows_gt_columns)
{
    T          query[1];
    const auto ldu = rows_gt_columns ? r : static_cast<lapack_int>(std::min(r, c));
    const auto ldv = rows_gt_columns ? static_cast<lapack_int>(std::min(r, c)) : c;
    if (rows_gt_columns)
    {
        gesvd_work(LAPACK_ROW_MAJOR,
            'O',
            'S',
            r,
            c,
            nullptr,
            ldu,
            nullptr,
            nullptr,
            ldu,
            nullptr,
            ldv,
            query,
            -1);
    }
    else
    {
        gesvd_work(LAPACK_ROW_MAJOR,
            'S',
            'O',
            r,
            c,
            nullptr,
            ldv,
            nullptr,
            nullptr,
            ldu,
            nullptr,
            ldv,
            query,
            -1);
    }
    return static_cast<lapack_int>(query[0]);
}

template <typename T, typename GesvdWorkFn>
size_t svd_workspace_size_impl(GesvdWorkFn gesvd_work, linalg_long rows, linalg_long columns)
{
    const auto              r = static_cast<lapack_int>(rows);
    const auto              c = static_cast<lapack_int>(columns);
    detail::workspace_arena arena;
    svd_plan<T>(arena, svd_lwork<T>(gesvd_work, r, c, rows > columns));
    return arena.total_size();
}

template <typename T, typename GesvdWorkFn>
void svd_lapacke_impl(GesvdWorkFn gesvd_work,
    const char*                   backend_name,
    linalg_long                   rows,
    linalg_long                   columns,
    T*                            A,
    linalg_long                   lda,
    T*                            S,
    T*                            U,
    linalg_long                   ldu,
    T*                            VT,
    linalg_long                   ldv,
    void*                         workspace,
    size_t                        workspace_size)
{
    const auto r     = static_cast<lapack_int>(rows);
    const auto c     = static_cast<lapack_int>(columns);
    const auto lwork = svd_lwork<T>(gesvd_work, r, c, rows > columns);

    LOGGING_CHECK(workspace_size >= svd_workspace_size_impl<T>(gesvd_work, rows, columns),
        "svd_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = svd_plan<T>(arena, lwork);

    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = gesvd_work(LAPACK_ROW_MAJOR,
            'O',
            'S',
            r,
            c,
            U,
            static_cast<lapack_int>(ldu),
            S,
            U,
            static_cast<lapack_int>(ldu),
            VT,
            static_cast<lapack_int>(ldv),
            bufs.work,
            lwork);
        LOGGING_CHECK(info == 0, "{} SVD was unsuccessful", backend_name);
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = gesvd_work(LAPACK_ROW_MAJOR,
            'S',
            'O',
            r,
            c,
            VT,
            static_cast<lapack_int>(ldv),
            S,
            U,
            static_cast<lapack_int>(ldu),
            VT,
            static_cast<lapack_int>(ldv),
            bufs.work,
            lwork);
        LOGGING_CHECK(info == 0, "{} SVD was unsuccessful", backend_name);
    }
}

}  // namespace

template <>
void svd_decomposition<float, linalg_backend::blas_lapacke>::apply(linalg_long rows,
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
    const auto r     = static_cast<lapack_int>(rows);
    const auto c     = static_cast<lapack_int>(columns);
    const auto lwork = svd_lwork<float>(LAPACKE_sgesvd_work, r, c, rows > columns);
    LOGGING_CHECK(
        workspace_size >= svd_workspace_size_impl<float>(LAPACKE_sgesvd_work, rows, columns),
        "svd_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = svd_plan<float>(arena, lwork);
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR,
            'O',
            'S',
            r,
            c,
            U,
            static_cast<lapack_int>(ldu),
            S,
            U,
            static_cast<lapack_int>(ldu),
            VT,
            static_cast<lapack_int>(ldv),
            bufs.work,
            lwork);
        LOGGING_CHECK(info == 0, "blas_lapack SVD was unsuccessful");
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR,
            'S',
            'O',
            r,
            c,
            VT,
            static_cast<lapack_int>(ldv),
            S,
            U,
            static_cast<lapack_int>(ldu),
            VT,
            static_cast<lapack_int>(ldv),
            bufs.work,
            lwork);
        LOGGING_CHECK(info == 0, "blas_lapack SVD was unsuccessful");
    }
}

template <>
void svd_decomposition<double, linalg_backend::blas_lapacke>::apply(linalg_long rows,
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
    const auto r     = static_cast<lapack_int>(rows);
    const auto c     = static_cast<lapack_int>(columns);
    const auto lwork = svd_lwork<double>(LAPACKE_dgesvd_work, r, c, rows > columns);
    LOGGING_CHECK(
        workspace_size >= svd_workspace_size_impl<double>(LAPACKE_dgesvd_work, rows, columns),
        "svd_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = svd_plan<double>(arena, lwork);
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR,
            'O',
            'S',
            r,
            c,
            U,
            static_cast<lapack_int>(ldu),
            S,
            U,
            static_cast<lapack_int>(ldu),
            VT,
            static_cast<lapack_int>(ldv),
            bufs.work,
            lwork);
        LOGGING_CHECK(info == 0, "blas_lapack SVD was unsuccessful");
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR,
            'S',
            'O',
            r,
            c,
            VT,
            static_cast<lapack_int>(ldv),
            S,
            U,
            static_cast<lapack_int>(ldu),
            VT,
            static_cast<lapack_int>(ldv),
            bufs.work,
            lwork);
        LOGGING_CHECK(info == 0, "blas_lapack SVD was unsuccessful");
    }
}

template <>
size_t svd_decomposition<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return svd_workspace_size_impl<float>(LAPACKE_sgesvd_work, rows, columns);
}

template <>
size_t svd_decomposition<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return svd_workspace_size_impl<double>(LAPACKE_dgesvd_work, rows, columns);
}

}  // namespace linalg
#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
