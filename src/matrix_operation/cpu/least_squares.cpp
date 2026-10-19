#include "include/matrix_operation/least_squares.h"

#include "include/common/workspace_arena.h"
#include "include/matrix_operation/pseudo_inverse.h"
#include "include/common/linalg_checks.h"

#include "include/common/linalg_profiler.h"

namespace linalg
{

template <typename T> struct least_squares_layout
{
    T*             Ainv;
    unsigned char* pinv_workspace;
};

template <typename T>
least_squares_layout<T> least_squares_plan(detail::workspace_arena& arena,
    linalg_long                                                     rows,
    linalg_long                                                     columns,
    size_t                                                          pinv_workspace_bytes)
{
    least_squares_layout<T> p;
    p.Ainv           = arena.next<T>(static_cast<size_t>(columns) * static_cast<size_t>(rows));
    p.pinv_workspace = arena.next<unsigned char>(pinv_workspace_bytes);
    return p;
}

template <typename T, linalg_backend Backend>
size_t least_squares_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto pinv_bytes =
        pseudo_inverse<T, Backend>::workspace_size(rows, columns);
    detail::workspace_arena arena;
    least_squares_plan<T>(arena, rows, columns, pinv_bytes);
    return arena.total_size();
}

template <typename T, linalg_backend Backend>
void least_squares_solve_impl(linalg_long rows,
    linalg_long                           columns,
    linalg_long                           nrhs,
    const T*                              A,
    linalg_long                           lda,
    const T*                              B,
    linalg_long                           ldb,
    T*                                    X,
    linalg_long                           ldx,
    void*                                 workspace,
    size_t                                workspace_size)
{
    const auto pinv_bytes =
        pseudo_inverse<T, Backend>::workspace_size(rows, columns);
    const size_t needed = least_squares_workspace_size_impl<T, Backend>(rows, columns);
    LINALG_CHECK(workspace_size >= needed, "least_squares_solve: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = least_squares_plan<T>(arena, rows, columns, pinv_bytes);

    // Ainv is columns x rows.
    linalg::pseudo_inverse<T, Backend>::apply(rows,
        columns,
        A,
        lda,
        bufs.Ainv,
        rows,
        static_cast<T>(-1),
        bufs.pinv_workspace,
        pinv_bytes);

    // X (columns x nrhs) = Ainv (columns x rows) * B (rows x nrhs).
    for (linalg_long i = 0; i < columns; ++i)
    {
        const T*       Ainv_row = bufs.Ainv + i * rows;
        T* const       X_row    = X + i * ldx;
        for (linalg_long j = 0; j < nrhs; ++j)
        {
            T sum = T(0);
            for (linalg_long k = 0; k < rows; ++k)
            {
                sum += Ainv_row[k] * B[k * ldb + j];
            }
            X_row[j] = sum;
        }
    }
}

template <>
void least_squares<float, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                              columns,
    linalg_long                                                              nrhs,
    const float*                                                             A,
    linalg_long                                                              lda,
    const float*                                                             B,
    linalg_long                                                              ldb,
    float*                                                                   X,
    linalg_long                                                              ldx,
    void*                                                                    workspace,
    size_t                                                                   workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("least_squares/native<float>");
    LINALG_CHECK(A != nullptr && B != nullptr && X != nullptr,
        "least_squares::apply: A, B, and X must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0,
        "least_squares::apply: rows/columns/nrhs/lda/ldb/ldx must be positive");
    least_squares_solve_impl<float, linalg_backend::native>(
        rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace, workspace_size);
}

template <>
void least_squares<double, linalg_backend::native>::apply(linalg_long rows,
    linalg_long                                                               columns,
    linalg_long                                                               nrhs,
    const double*                                                             A,
    linalg_long                                                               lda,
    const double*                                                             B,
    linalg_long                                                               ldb,
    double*                                                                   X,
    linalg_long                                                               ldx,
    void*                                                                     workspace,
    size_t                                                                    workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("least_squares/native<double>");
    LINALG_CHECK(A != nullptr && B != nullptr && X != nullptr,
        "least_squares::apply: A, B, and X must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0,
        "least_squares::apply: rows/columns/nrhs/lda/ldb/ldx must be positive");
    least_squares_solve_impl<double, linalg_backend::native>(
        rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace, workspace_size);
}

template <>
size_t least_squares<float, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return least_squares_workspace_size_impl<float, linalg_backend::native>(rows, columns);
}

template <>
size_t least_squares<double, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return least_squares_workspace_size_impl<double, linalg_backend::native>(rows, columns);
}

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

template <>
void least_squares<float, linalg_backend::blas_lapacke>::apply(linalg_long rows,
    linalg_long                                                                       columns,
    linalg_long                                                                       nrhs,
    const float*                                                                      A,
    linalg_long                                                                       lda,
    const float*                                                                      B,
    linalg_long                                                                       ldb,
    float*                                                                            X,
    linalg_long                                                                       ldx,
    void*                                                                             workspace,
    size_t                                                                            workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("least_squares/blas_lapacke<float>");
    LINALG_CHECK(A != nullptr && B != nullptr && X != nullptr,
        "least_squares::apply: A, B, and X must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0,
        "least_squares::apply: rows/columns/nrhs/lda/ldb/ldx must be positive");
    least_squares_solve_impl<float, linalg_backend::blas_lapacke>(
        rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace, workspace_size);
}

template <>
void least_squares<double, linalg_backend::blas_lapacke>::apply(linalg_long rows,
    linalg_long                                                                        columns,
    linalg_long                                                                        nrhs,
    const double*                                                                      A,
    linalg_long                                                                        lda,
    const double*                                                                      B,
    linalg_long                                                                        ldb,
    double*                                                                            X,
    linalg_long                                                                        ldx,
    void*                                                                              workspace,
    size_t                                                                             workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("least_squares/blas_lapacke<double>");
    LINALG_CHECK(A != nullptr && B != nullptr && X != nullptr,
        "least_squares::apply: A, B, and X must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0,
        "least_squares::apply: rows/columns/nrhs/lda/ldb/ldx must be positive");
    least_squares_solve_impl<double, linalg_backend::blas_lapacke>(
        rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace, workspace_size);
}

template <>
size_t least_squares<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return least_squares_workspace_size_impl<float, linalg_backend::blas_lapacke>(rows, columns);
}

template <>
size_t least_squares<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return least_squares_workspace_size_impl<double, linalg_backend::blas_lapacke>(rows, columns);
}

#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#if LINALG_HAS_MKL

template <>
void least_squares<float, linalg_backend::mkl>::apply(linalg_long rows,
    linalg_long                                                              columns,
    linalg_long                                                              nrhs,
    const float*                                                             A,
    linalg_long                                                              lda,
    const float*                                                             B,
    linalg_long                                                              ldb,
    float*                                                                   X,
    linalg_long                                                              ldx,
    void*                                                                    workspace,
    size_t                                                                   workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("least_squares/mkl<float>");
    LINALG_CHECK(A != nullptr && B != nullptr && X != nullptr,
        "least_squares::apply: A, B, and X must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0,
        "least_squares::apply: rows/columns/nrhs/lda/ldb/ldx must be positive");
    least_squares_solve_impl<float, linalg_backend::mkl>(
        rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace, workspace_size);
}

template <>
void least_squares<double, linalg_backend::mkl>::apply(linalg_long rows,
    linalg_long                                                               columns,
    linalg_long                                                               nrhs,
    const double*                                                             A,
    linalg_long                                                               lda,
    const double*                                                             B,
    linalg_long                                                               ldb,
    double*                                                                   X,
    linalg_long                                                               ldx,
    void*                                                                     workspace,
    size_t                                                                    workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("least_squares/mkl<double>");
    LINALG_CHECK(A != nullptr && B != nullptr && X != nullptr,
        "least_squares::apply: A, B, and X must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0,
        "least_squares::apply: rows/columns/nrhs/lda/ldb/ldx must be positive");
    least_squares_solve_impl<double, linalg_backend::mkl>(
        rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace, workspace_size);
}

template <>
size_t least_squares<float, linalg_backend::mkl>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return least_squares_workspace_size_impl<float, linalg_backend::mkl>(rows, columns);
}

template <>
size_t least_squares<double, linalg_backend::mkl>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return least_squares_workspace_size_impl<double, linalg_backend::mkl>(rows, columns);
}

#endif  // LINALG_HAS_MKL

}  // namespace linalg
