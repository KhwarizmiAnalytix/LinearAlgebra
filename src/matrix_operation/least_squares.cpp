#include "include/matrix_operation/least_squares.h"

#include "include/common/workspace_arena.h"
#include "include/matrix_operation/pseudo_inverse.h"
#include <include/logging.h>

namespace linalg
{
namespace detail
{

template <typename T>
struct least_squares_layout
{
    T*             Ainv;
    unsigned char* pinv_workspace;
};

template <typename T>
least_squares_layout<T> least_squares_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns, size_t pinv_workspace_bytes)
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
        pseudo_inverse_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    detail::workspace_arena arena;
    least_squares_plan<T>(arena, rows, columns, pinv_bytes);
    return arena.total_size();
}

template <typename T>
void least_squares_solve_impl(linalg_long rows,
    linalg_long                            columns,
    linalg_long                            nrhs,
    const T*                                 A,
    linalg_long                            lda,
    const T*                                 B,
    linalg_long                            ldb,
    T*                                       X,
    linalg_long                            ldx,
    void*                                    workspace,
    size_t                                   workspace_size)
{
    const auto pinv_bytes =
        pseudo_inverse_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    LOGGING_CHECK(workspace_size >= least_squares_workspace_size_impl<T>(rows, columns),
        "least_squares_solve: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = least_squares_plan<T>(arena, rows, columns, pinv_bytes);

    // Ainv is columns x rows.
    linalg::pseudo_inverse(rows, columns, A, lda, bufs.Ainv, rows, static_cast<T>(-1),
        bufs.pinv_workspace, pinv_bytes);

    // X (columns x nrhs) = Ainv (columns x rows) * B (rows x nrhs).
    for (linalg_long i = 0; i < columns; ++i)
    {
        for (linalg_long j = 0; j < nrhs; ++j)
        {
            T sum = T(0);
            for (linalg_long k = 0; k < rows; ++k)
            {
                sum += bufs.Ainv[i * rows + k] * B[k * ldb + j];
            }
            X[i * ldx + j] = sum;
        }
    }
}

}  // namespace detail

size_t least_squares_solve_workspace_size(const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return detail::least_squares_workspace_size_impl<float>(rows, columns);
}

size_t least_squares_solve_workspace_size(const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return detail::least_squares_workspace_size_impl<double>(rows, columns);
}

//-----------------------------------------------------------------------------
void least_squares_solve(linalg_long rows,
    linalg_long                      columns,
    linalg_long                      nrhs,
    const float*                       A,
    linalg_long                      lda,
    const float*                       B,
    linalg_long                      ldb,
    float*                             X,
    linalg_long                      ldx,
    void*                              workspace,
    size_t                             workspace_size)
{
    LOGGING_CHECK(A != nullptr && B != nullptr && X != nullptr, "least_squares_solve: A, B, and X must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0, "least_squares_solve: rows/columns/nrhs/lda/ldb/ldx must be positive");
    detail::least_squares_solve_impl(rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace, workspace_size);
}

//-----------------------------------------------------------------------------
void least_squares_solve(linalg_long rows,
    linalg_long                      columns,
    linalg_long                      nrhs,
    const double*                      A,
    linalg_long                      lda,
    const double*                      B,
    linalg_long                      ldb,
    double*                            X,
    linalg_long                      ldx,
    void*                              workspace,
    size_t                             workspace_size)
{
    LOGGING_CHECK(A != nullptr && B != nullptr && X != nullptr, "least_squares_solve: A, B, and X must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0, "least_squares_solve: rows/columns/nrhs/lda/ldb/ldx must be positive");
    detail::least_squares_solve_impl(rows, columns, nrhs, A, lda, B, ldb, X, ldx, workspace, workspace_size);
}

}  // namespace linalg
