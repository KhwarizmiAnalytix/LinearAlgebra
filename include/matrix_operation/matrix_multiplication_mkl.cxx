#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_MKL

#include <mkl.h>

#include "include/matrix_operation/matrix_multiplication_dispatch.h"

namespace linalg
{
namespace detail
{

void matmul_mkl_f32(
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
    quarisma_int ldc)
{
    cblas_sgemm(
        CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        rows,
        columns,
        depth,
        1.F,
        a,
        lda,
        b,
        ldb,
        0.F,
        c,
        ldc);
}

void matmul_mkl_f64(
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
    quarisma_int  ldc)
{
    cblas_dgemm(
        CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        rows,
        columns,
        depth,
        1.,
        a,
        lda,
        b,
        ldb,
        0.,
        c,
        ldc);
}

LINALG_REGISTER_DISPATCH(matmul_f32_stub, mkl, matmul_mkl_f32);
LINALG_REGISTER_DISPATCH(matmul_f64_stub, mkl, matmul_mkl_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_MKL
