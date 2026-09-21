#include "include/matrix_operation/least_squares.h"

#include "include/matrix_operation/pseudo_inverse.h"
#include "include/memory/allocator.h"
#include "ThirdParty/Logging/include/logging.h"

namespace linalg
{
namespace detail
{

template <typename T>
void least_squares_solve_impl(linalg_long rows,
    linalg_long                            columns,
    linalg_long                            nrhs,
    const T*                                 A,
    linalg_long                            lda,
    const T*                                 B,
    linalg_long                            ldb,
    T*                                       X,
    linalg_long                            ldx)
{
    using Allocator = linalg::allocator<T>;

    // Ainv is columns x rows.
    auto* Ainv = Allocator::allocate(columns * rows);
    linalg::pseudo_inverse(rows, columns, A, lda, Ainv, rows);

    // X (columns x nrhs) = Ainv (columns x rows) * B (rows x nrhs).
    for (linalg_long i = 0; i < columns; ++i)
    {
        for (linalg_long j = 0; j < nrhs; ++j)
        {
            T sum = T(0);
            for (linalg_long k = 0; k < rows; ++k)
            {
                sum += Ainv[i * rows + k] * B[k * ldb + j];
            }
            X[i * ldx + j] = sum;
        }
    }

    Allocator::free(Ainv);
}

}  // namespace detail

//-----------------------------------------------------------------------------
void least_squares_solve(linalg_long rows,
    linalg_long                      columns,
    linalg_long                      nrhs,
    const float*                       A,
    linalg_long                      lda,
    const float*                       B,
    linalg_long                      ldb,
    float*                             X,
    linalg_long                      ldx)
{
    LOGGING_CHECK(A != nullptr && B != nullptr && X != nullptr, "least_squares_solve: A, B, and X must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0, "least_squares_solve: rows/columns/nrhs/lda/ldb/ldx must be positive");
    detail::least_squares_solve_impl(rows, columns, nrhs, A, lda, B, ldb, X, ldx);
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
    linalg_long                      ldx)
{
    LOGGING_CHECK(A != nullptr && B != nullptr && X != nullptr, "least_squares_solve: A, B, and X must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0 && nrhs != 0 && lda != 0 && ldb != 0 && ldx != 0, "least_squares_solve: rows/columns/nrhs/lda/ldb/ldx must be positive");
    detail::least_squares_solve_impl(rows, columns, nrhs, A, lda, B, ldb, X, ldx);
}

}  // namespace linalg
