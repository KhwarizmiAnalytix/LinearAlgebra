#include "include/matrix_operation_gpu/multiplication_gpu.h"

#if LINALG_HAS_CUBLAS

#include <cublas_v2.h>

#include "include/common/cuda_handle.h"
#include "include/logging.h"
#include "include/common/narrow.h"

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
void multiplication(bool transpose_a,
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
    LOGGING_CHECK(a != nullptr && b != nullptr && c != nullptr, "multiplication: a, b, and c must not be null");
    LOGGING_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0, "multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
    const float alpha  = 1.F;
    const float beta   = 0.F;
    auto        handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);
    if (cublasSgemm(handle,
            transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
            transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
            detail::checked_narrow<int>(columns),
            detail::checked_narrow<int>(rows),
            detail::checked_narrow<int>(depth),
            &alpha,
            b,
            detail::checked_narrow<int>(ldb),
            a,
            detail::checked_narrow<int>(lda),
            &beta,
            c,
            detail::checked_narrow<int>(ldc)) != CUBLAS_STATUS_SUCCESS)
    {
        LOGGING_THROW("cublasSgemm failed");
    }
}

void multiplication(bool transpose_a,
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
    LOGGING_CHECK(a != nullptr && b != nullptr && c != nullptr, "multiplication: a, b, and c must not be null");
    LOGGING_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0, "multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
    const double alpha  = 1.;
    const double beta   = 0.;
    auto         handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);
    if (cublasDgemm(handle,
            transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
            transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
            detail::checked_narrow<int>(columns),
            detail::checked_narrow<int>(rows),
            detail::checked_narrow<int>(depth),
            &alpha,
            b,
            detail::checked_narrow<int>(ldb),
            a,
            detail::checked_narrow<int>(lda),
            &beta,
            c,
            detail::checked_narrow<int>(ldc)) != CUBLAS_STATUS_SUCCESS)
    {
        LOGGING_THROW("cublasDgemm failed");
    }
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
