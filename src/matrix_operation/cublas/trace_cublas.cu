#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>

#include "include/matrix_operation/trace.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

// Single-thread diagonal sum: n is one matrix dimension, not a large array,
// so a parallel reduction buys nothing (same rationale as
// lu_determinant_kernel in inversion_cublas.cu). `A` is row-major n x n,
// tightly packed (lda == n), so the diagonal offset i*n+i is unambiguous.
template <typename scalar_t>
static __global__ void trace_kernel(const scalar_t* A, int n, scalar_t* out)
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
namespace
{

template <typename T>
size_t trace_workspace_size_impl()
{
    detail::workspace_arena arena;
    arena.next<T>(1);
    return arena.total_size();
}

template <typename T>
T trace_impl(const T* A, linalg_int n, linalg_int lda, void* stream)
{
    if (A == nullptr)
    {
        LINALG_THROW("trace: A must not be null");
    }
    if (n <= 0)
    {
        LINALG_THROW("trace: n must be positive");
    }
    if (lda != n)
    {
        LINALG_THROW("trace<cuda_blas>: requires tightly packed lda (lda == n)");
    }
    const auto stream_t = static_cast<cudaStream_t>(stream);

    // The struct interface has no workspace parameter; allocate internally.
    const auto ws_size = trace_workspace_size_impl<T>();
    void*      workspace = nullptr;
    LINALG_CHECK(cudaMalloc(&workspace, ws_size) == cudaSuccess,
        "trace<cuda_blas>: cudaMalloc for workspace failed");

    detail::workspace_arena arena(workspace, ws_size);
    T*                       dev_sum = arena.next<T>(1);

    trace_kernel<T><<<1, 1, 0, stream_t>>>(A, detail::checked_narrow<int>(n), dev_sum);
    const auto launch_err = cudaGetLastError();

    T          sum         = T(0);
    const auto copy_status =
        cudaMemcpyAsync(&sum, dev_sum, sizeof(T), cudaMemcpyDeviceToHost, stream_t);
    const auto sync_status = cudaStreamSynchronize(stream_t);
    cudaFree(workspace);

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

template <>
float trace<float, linalg_backend::cuda_blas>::apply(
    const float* A, linalg_int n, linalg_int lda, void* stream)
{
    LINALG_SCOPE("trace/cuda_blas<float>");
    return trace_impl(A, n, lda, stream);
}

template <>
double trace<double, linalg_backend::cuda_blas>::apply(
    const double* A, linalg_int n, linalg_int lda, void* stream)
{
    LINALG_SCOPE("trace/cuda_blas<double>");
    return trace_impl(A, n, lda, stream);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
