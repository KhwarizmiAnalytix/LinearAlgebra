#if LINALG_HAS_ROCBLAS

#include <hipblas/hipblas.h>

#include "include/matrix_operation/multiplication.h"
#include "include/common/hip_handle.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace
{

// hipBLAS is column-major; a/b/c here are row-major device buffers (same
// convention as the CPU backends). Rather than transposing data on the
// device, this uses the standard row-major-via-column-major identity:
// row-major C[m,n] = op(A)[m,k]*op(B)[k,n]  <=>  column-major
// C^T[n,m] = op(B)^T[n,k]*op(A)^T[k,m], computed by swapping the A/B
// operands (and m/n) while keeping each operand's own transpose flag and
// leading dimension — no extra copies or kernels needed.
template <typename T, typename GemmFn>
void multiplication_impl(GemmFn gemm,
    const T  alpha,
    const T  beta,
    bool     transpose_a,
    bool     transpose_b,
    linalg_int rows,
    linalg_int columns,
    linalg_int depth,
    const T*   a,
    linalg_int lda,
    const T*   b,
    linalg_int ldb,
    T*         c,
    linalg_int ldc,
    hipStream_t stream)
{
    LINALG_CHECK(a != nullptr && b != nullptr && c != nullptr,
        "multiplication: a, b, and c must not be null");
    LINALG_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0,
        "multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
    auto handle = detail::hipblas_handle_for_current_device();
    detail::set_stream(handle, stream);
    if (gemm(handle,
            transpose_b ? HIPBLAS_OP_T : HIPBLAS_OP_N,
            transpose_a ? HIPBLAS_OP_T : HIPBLAS_OP_N,
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
            detail::checked_narrow<int>(ldc)) != HIPBLAS_STATUS_SUCCESS)
    {
        LINALG_THROW("hipblasSgemm/Dgemm failed");
    }
}

}  // namespace

template <>
void multiplication<float, linalg_backend::ROCM_BLAS>::apply(bool transpose_a,
    bool           transpose_b,
    linalg_int     rows,
    linalg_int     columns,
    linalg_int     depth,
    const float*   a,
    linalg_int     lda,
    const float*   b,
    linalg_int     ldb,
    float*         c,
    linalg_int     ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void*          stream)
{
    LINALG_SCOPE("multiplication/rocm_blas<float>");
    const float alpha = 1.F;
    const float beta  = 0.F;
    multiplication_impl(hipblasSgemm, alpha, beta, transpose_a, transpose_b,
        rows, columns, depth, a, lda, b, ldb, c, ldc,
        static_cast<hipStream_t>(stream));
}

template <>
void multiplication<double, linalg_backend::ROCM_BLAS>::apply(bool transpose_a,
    bool           transpose_b,
    linalg_int     rows,
    linalg_int     columns,
    linalg_int     depth,
    const double*  a,
    linalg_int     lda,
    const double*  b,
    linalg_int     ldb,
    double*        c,
    linalg_int     ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void*          stream)
{
    LINALG_SCOPE("multiplication/rocm_blas<double>");
    const double alpha = 1.;
    const double beta  = 0.;
    multiplication_impl(hipblasDgemm, alpha, beta, transpose_a, transpose_b,
        rows, columns, depth, a, lda, b, ldb, c, ldc,
        static_cast<hipStream_t>(stream));
}

template <>
size_t multiplication<float, linalg_backend::ROCM_BLAS>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

template <>
size_t multiplication<double, linalg_backend::ROCM_BLAS>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
