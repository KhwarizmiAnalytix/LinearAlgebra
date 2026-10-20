#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>

#include "include/matrix_operation/least_squares.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/pseudo_inverse.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

// Least-squares via pseudo-inverse: X = pinv(A) * B.
// Calls the struct API for pseudo_inverse and multiplication directly so that
// no cross-TU symbol other than detail::svd_on_device (already reachable
// through pseudo_inverse_cublas.cu) is needed.
namespace linalg
{
namespace
{

template <typename T>
struct least_squares_layout
{
    T*             Ainv;
    unsigned char* pinv_workspace;
};

template <typename T>
least_squares_layout<T> least_squares_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns,
    size_t pinv_workspace_bytes)
{
    least_squares_layout<T> p;
    p.Ainv           = arena.next<T>(static_cast<size_t>(columns) * static_cast<size_t>(rows));
    p.pinv_workspace = arena.next<unsigned char>(pinv_workspace_bytes);
    return p;
}

template <typename T>
size_t least_squares_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto pinv_bytes =
        pseudo_inverse<T, linalg_backend::CUDA_BLAS>::workspace_size(rows, columns);
    detail::workspace_arena arena;
    least_squares_plan<T>(arena, rows, columns, pinv_bytes);
    return arena.total_size();
}

template <typename T>
void least_squares_solve_impl(linalg_long rows,
    linalg_long                            columns,
    linalg_long                            nrhs,
    const T*                               A,
    linalg_long                            lda,
    const T*                               B,
    linalg_long                            ldb,
    T*                                     X,
    linalg_long                            ldx,
    void*                                  workspace,
    size_t                                 workspace_size,
    void*                                  stream)
{
    LINALG_CHECK(A != nullptr && B != nullptr && X != nullptr,
        "least_squares: A, B, and X must not be null");
    LINALG_CHECK(rows != 0 && columns != 0 && nrhs != 0,
        "least_squares: rows/columns/nrhs must be positive");
    LINALG_CHECK(!(lda != columns) && !(ldb != nrhs) && !(ldx != nrhs),
        "least_squares<cuda_blas>: requires tightly packed lda/ldb/ldx "
        "(lda == columns, ldb == nrhs, ldx == nrhs)");

    const auto pinv_bytes =
        pseudo_inverse<T, linalg_backend::CUDA_BLAS>::workspace_size(rows, columns);
    LINALG_CHECK(workspace_size >= least_squares_workspace_size_impl<T>(rows, columns),
        "least_squares: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = least_squares_plan<T>(arena, rows, columns, pinv_bytes);

    pseudo_inverse<T, linalg_backend::CUDA_BLAS>::apply(rows, columns, A, lda, bufs.Ainv, rows,
        static_cast<T>(-1), bufs.pinv_workspace, pinv_bytes, stream);

    // X (columns x nrhs) = Ainv (columns x rows) * B (rows x nrhs).
    multiplication<T, linalg_backend::CUDA_BLAS>::apply(
        false, false, static_cast<linalg_int>(columns), static_cast<linalg_int>(nrhs),
        static_cast<linalg_int>(rows), bufs.Ainv, static_cast<linalg_int>(rows), B,
        static_cast<linalg_int>(ldb), X, static_cast<linalg_int>(ldx), nullptr, 0, stream);
}

}  // namespace

template <>
void least_squares<float, linalg_backend::CUDA_BLAS>::apply(linalg_long rows,
    linalg_long   columns,
    linalg_long   nrhs,
    const float*  A,
    linalg_long   lda,
    const float*  B,
    linalg_long   ldb,
    float*        X,
    linalg_long   ldx,
    void*         workspace,
    size_t        workspace_size,
    void*         stream)
{
    LINALG_SCOPE("least_squares/cuda_blas<float>");
    least_squares_solve_impl(rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace,
        workspace_size, stream);
}

template <>
void least_squares<double, linalg_backend::CUDA_BLAS>::apply(linalg_long rows,
    linalg_long    columns,
    linalg_long    nrhs,
    const double*  A,
    linalg_long    lda,
    const double*  B,
    linalg_long    ldb,
    double*        X,
    linalg_long    ldx,
    void*          workspace,
    size_t         workspace_size,
    void*          stream)
{
    LINALG_SCOPE("least_squares/cuda_blas<double>");
    least_squares_solve_impl(rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace,
        workspace_size, stream);
}

template <>
size_t least_squares<float, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return least_squares_workspace_size_impl<float>(rows, columns);
}

template <>
size_t least_squares<double, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return least_squares_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
