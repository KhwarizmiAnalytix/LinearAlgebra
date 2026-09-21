#include "include/matrix_operation_gpu/least_squares_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation_gpu/matrix_multiplication_gpu.h"
#include "include/matrix_operation_gpu/pseudo_inverse_gpu.h"
#include "include/util/exception.h"

namespace linalg
{
namespace gpu
{
namespace
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
    linalg_long                            ldx,
    cudaStream_t                             stream)
{
    if (A == nullptr || B == nullptr || X == nullptr)
    {
        LINALG_THROW("least_squares_solve: A, B, and X must not be null");
    }
    if (rows == 0 || columns == 0 || nrhs == 0)
    {
        LINALG_THROW("least_squares_solve: rows/columns/nrhs must be positive");
    }
    if (lda != columns || ldb != nrhs || ldx != nrhs)
    {
        LINALG_THROW("linalg::gpu::least_squares_solve requires tightly packed lda/ldb/ldx (lda == "
                     "columns, ldb == nrhs, ldx == nrhs)");
    }

    T* Ainv = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&Ainv),
            sizeof(T) * static_cast<size_t>(columns) * static_cast<size_t>(rows)) != cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }

    pseudo_inverse(rows, columns, A, lda, Ainv, rows, static_cast<T>(-1), stream);

    // X (columns x nrhs) = Ainv (columns x rows) * B (rows x nrhs).
    matrix_multiplication(false,
        false,
        static_cast<linalg_int>(columns),
        static_cast<linalg_int>(nrhs),
        static_cast<linalg_int>(rows),
        Ainv,
        static_cast<linalg_int>(rows),
        B,
        static_cast<linalg_int>(ldb),
        X,
        static_cast<linalg_int>(ldx),
        stream);

    // Ainv is still being read by the GEMM just launched above; wait for
    // `stream` before freeing it (see pseudo_inverse_gpu.cu's identical
    // comment for why a bare cudaFree here is not safe in general).
    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(Ainv, stream, &sync_status);
    if (sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaStreamSynchronize failed");
    }
}

}  // namespace

void least_squares_solve(linalg_long rows,
    linalg_long                      columns,
    linalg_long                      nrhs,
    const float*                       A,
    linalg_long                      lda,
    const float*                       B,
    linalg_long                      ldb,
    float*                             X,
    linalg_long                      ldx,
    cudaStream_t                       stream)
{
    least_squares_solve_impl(rows, columns, nrhs, A, lda, B, ldb, X, ldx, stream);
}

void least_squares_solve(linalg_long rows,
    linalg_long                      columns,
    linalg_long                      nrhs,
    const double*                      A,
    linalg_long                      lda,
    const double*                      B,
    linalg_long                      ldb,
    double*                            X,
    linalg_long                      ldx,
    cudaStream_t                       stream)
{
    least_squares_solve_impl(rows, columns, nrhs, A, lda, B, ldb, X, ldx, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
