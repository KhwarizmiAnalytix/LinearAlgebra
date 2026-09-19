#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_BLAS

#ifdef LINALG_BLAS_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include "include/matrix_operation/matrix_multiplication_dispatch.h"

namespace linalg
{
namespace detail
{

void matmul_blas_f32(
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
        static_cast<int>(rows),
        static_cast<int>(columns),
        static_cast<int>(depth),
        1.F,
        a,
        static_cast<int>(lda),
        b,
        static_cast<int>(ldb),
        0.F,
        c,
        static_cast<int>(ldc));
}

void matmul_blas_f64(
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
        static_cast<int>(rows),
        static_cast<int>(columns),
        static_cast<int>(depth),
        1.,
        a,
        static_cast<int>(lda),
        b,
        static_cast<int>(ldb),
        0.,
        c,
        static_cast<int>(ldc));
}

LINALG_REGISTER_DISPATCH(matmul_f32_stub, blas, matmul_blas_f32);
LINALG_REGISTER_DISPATCH(matmul_f64_stub, blas, matmul_blas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_BLAS
