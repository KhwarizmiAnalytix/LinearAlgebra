#include "include/matrix_operation_gpu/matrix_multiplication_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cublas_v2.h>

#include "include/common/cuda_handle.h"
#include "include/util/exception.h"

namespace linalg
{
namespace gpu
{

// cuBLAS is column-major; a/b/c here are row-major device buffers (same
// convention as the CPU backends). Rather than transposing data on the
// device, this uses the standard row-major-via-column-major identity:
// row-major C[m,n] = op(A)[m,k]*op(B)[k,n]  <=>  column-major
// C^T[n,m] = op(B)^T[n,k]*op(A)^T[k,m], computed by swapping the A/B
// operands (and m/n) while keeping each operand's own transpose flag and
// leading dimension — no extra copies or kernels needed.
void matrix_multiplication(bool transpose_a,
    bool                        transpose_b,
    linalg_int                rows,
    linalg_int                columns,
    linalg_int                depth,
    float const*                a,
    linalg_int                lda,
    float const*                b,
    linalg_int                ldb,
    float*                      c,
    linalg_int                ldc,
    cudaStream_t                stream)
{
    if (a == nullptr || b == nullptr || c == nullptr)
    {
        LINALG_THROW("matrix_multiplication: a, b, and c must not be null");
    }
    if (rows <= 0 || columns <= 0 || depth <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0)
    {
        LINALG_THROW("matrix_multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
    }
    const float alpha  = 1.F;
    const float beta   = 0.F;
    auto        handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);
    if (cublasSgemm(handle,
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

void matrix_multiplication(bool transpose_a,
    bool                        transpose_b,
    linalg_int                rows,
    linalg_int                columns,
    linalg_int                depth,
    double const*               a,
    linalg_int                lda,
    double const*               b,
    linalg_int                ldb,
    double*                     c,
    linalg_int                ldc,
    cudaStream_t                stream)
{
    if (a == nullptr || b == nullptr || c == nullptr)
    {
        LINALG_THROW("matrix_multiplication: a, b, and c must not be null");
    }
    if (rows <= 0 || columns <= 0 || depth <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0)
    {
        LINALG_THROW("matrix_multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
    }
    const double alpha  = 1.;
    const double beta   = 0.;
    auto         handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);
    if (cublasDgemm(handle,
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

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
