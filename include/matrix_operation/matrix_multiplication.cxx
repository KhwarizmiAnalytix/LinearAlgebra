#include "include/matrix_operation/matrix_multiplication.h"

#include "include/matrix_operation/matrix_multiplication_dispatch.h"

namespace linalg
{
namespace detail
{

LINALG_DEFINE_DISPATCH(matmul_f32_fn, matmul_f32_stub);
LINALG_DEFINE_DISPATCH(matmul_f64_fn, matmul_f64_stub);

}  // namespace detail

void matrix_multiplication(
    bool         transpose_a,
    bool         transpose_b,
    quarisma_int rows,
    quarisma_int columns,
    quarisma_int depth,
    float const* a,
    quarisma_int lda,
    float const* b,
    quarisma_int ldb,
    float*       c,
    quarisma_int ldc,
    device_type  device)
{
    detail::matmul_f32_stub.resolve(device)(
        transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
}

void matrix_multiplication(
    bool          transpose_a,
    bool          transpose_b,
    quarisma_int  rows,
    quarisma_int  columns,
    quarisma_int  depth,
    double const* a,
    quarisma_int  lda,
    double const* b,
    quarisma_int  ldb,
    double*       c,
    quarisma_int  ldc,
    device_type   device)
{
    detail::matmul_f64_stub.resolve(device)(
        transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
}

}  // namespace linalg
