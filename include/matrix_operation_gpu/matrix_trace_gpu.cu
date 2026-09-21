#include "include/matrix_operation_gpu/matrix_trace_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>

#include "include/util/exception.h"

// Single-thread reduction: n is one matrix dimension, not a large array, so
// a parallel reduction buys nothing here — same rationale and shape as
// matrix_inversion_gpu.cu's lu_determinant_kernel/cholesky_determinant_
// kernel. `A` is row-major n x n, tightly packed (lda == n), so the
// diagonal offset i*n+i is unambiguous.
template <typename scalar_t> static __global__ void trace_kernel(const scalar_t* A, int n, scalar_t* out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    scalar_t sum = scalar_t(0);
    for (int i = 0; i < n; ++i)
    {
        sum += A[i * n + i];
    }
    *out = sum;
}

namespace linalg
{
namespace gpu
{
namespace
{

template <typename T> T matrix_trace_impl(const T* A, linalg_int n, linalg_int lda, cudaStream_t stream)
{
    if (A == nullptr)
    {
        LINALG_THROW("matrix_trace: A must not be null");
    }
    if (n <= 0)
    {
        LINALG_THROW("matrix_trace: n must be positive");
    }
    if (lda != n)
    {
        LINALG_THROW("linalg::gpu::matrix_trace requires tightly packed lda (lda == n)");
    }

    T* dev_sum = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_sum), sizeof(T)) != cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }

    trace_kernel<T><<<1, 1, 0, stream>>>(A, static_cast<int>(n), dev_sum);
    const auto launch_err = cudaGetLastError();

    T          sum = T(0);
    const auto copy_status = cudaMemcpyAsync(&sum, dev_sum, sizeof(T), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);
    cudaFree(dev_sum);

    if (launch_err != cudaSuccess)
    {
        LINALG_THROW("trace_kernel launch failed");
    }
    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return sum;
}

}  // namespace

float matrix_trace(const float* A, linalg_int n, linalg_int lda, cudaStream_t stream)
{
    return matrix_trace_impl(A, n, lda, stream);
}

double matrix_trace(const double* A, linalg_int n, linalg_int lda, cudaStream_t stream)
{
    return matrix_trace_impl(A, n, lda, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
