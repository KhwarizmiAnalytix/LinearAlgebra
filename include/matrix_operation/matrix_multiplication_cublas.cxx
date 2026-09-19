#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_CUBLAS

#include <cublas_v2.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation/matrix_multiplication_dispatch.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{

// cuBLAS is column-major; a/b/c here are row-major device buffers (same
// convention as the scalar/mkl/blas backends). Rather than transposing data
// on the device, this uses the standard row-major-via-column-major identity:
// row-major C[m,n] = op(A)[m,k]*op(B)[k,n]  <=>  column-major
// C^T[n,m] = op(B)^T[n,k]*op(A)^T[k,m], computed by swapping the A/B
// operands (and m/n) while keeping each operand's own transpose flag and
// leading dimension — no extra copies or kernels needed.
void matmul_cublas_f32(
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
    const float alpha = 1.F;
    const float beta  = 0.F;
    if (cublasSgemm(
            cublas_handle(),
            transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
            transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
            static_cast<int>(columns),
            static_cast<int>(rows),
            static_cast<int>(depth),
            &alpha,
            b,
            static_cast<int>(ldb),
            a,
            static_cast<int>(lda),
            &beta,
            c,
            static_cast<int>(ldc)) != CUBLAS_STATUS_SUCCESS)
    {
        LINALG_THROW("cublasSgemm failed");
    }
}

void matmul_cublas_f64(
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
    const double alpha = 1.;
    const double beta  = 0.;
    if (cublasDgemm(
            cublas_handle(),
            transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
            transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
            static_cast<int>(columns),
            static_cast<int>(rows),
            static_cast<int>(depth),
            &alpha,
            b,
            static_cast<int>(ldb),
            a,
            static_cast<int>(lda),
            &beta,
            c,
            static_cast<int>(ldc)) != CUBLAS_STATUS_SUCCESS)
    {
        LINALG_THROW("cublasDgemm failed");
    }
}

LINALG_REGISTER_DISPATCH(matmul_f32_stub, cuda, matmul_cublas_f32);
LINALG_REGISTER_DISPATCH(matmul_f64_stub, cuda, matmul_cublas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
