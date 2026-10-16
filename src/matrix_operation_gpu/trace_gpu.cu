#include "include/matrix_operation_gpu/trace_gpu.h"

#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>

#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"

// Single-thread reduction: n is one matrix dimension, not a large array, so
// a parallel reduction buys nothing here — same rationale and shape as
// inversion_gpu.cu's lu_determinant_kernel/cholesky_determinant_
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

template <typename T> size_t matrix_trace_workspace_size_impl()
{
    detail::workspace_arena arena;
    arena.next<T>(1);
    return arena.total_size();
}

template <typename T>
T matrix_trace_impl(const T* A, linalg_int n, linalg_int lda, void* workspace,
    size_t workspace_size, cudaStream_t stream)
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
        LINALG_THROW("linalg::gpu::trace requires tightly packed lda (lda == n)");
    }
    LINALG_CHECK(workspace_size >= matrix_trace_workspace_size_impl<T>(),
        "trace: workspace_size too small");

    detail::workspace_arena arena(workspace, workspace_size);
    T*                               dev_sum = arena.next<T>(1);

    trace_kernel<T><<<1, 1, 0, stream>>>(A, detail::checked_narrow<int>(n), dev_sum);
    const auto launch_err = cudaGetLastError();

    T          sum = T(0);
    const auto copy_status = cudaMemcpyAsync(&sum, dev_sum, sizeof(T), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);

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

size_t matrix_trace_workspace_size(const float* /*type_tag*/)
{
    return matrix_trace_workspace_size_impl<float>();
}

size_t matrix_trace_workspace_size(const double* /*type_tag*/)
{
    return matrix_trace_workspace_size_impl<double>();
}

float trace(const float* A, linalg_int n, linalg_int lda, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    return matrix_trace_impl(A, n, lda, workspace, workspace_size, stream);
}

double trace(const double* A, linalg_int n, linalg_int lda, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    return matrix_trace_impl(A, n, lda, workspace, workspace_size, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
