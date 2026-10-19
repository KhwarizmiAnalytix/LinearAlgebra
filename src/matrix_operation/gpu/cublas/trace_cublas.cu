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
void trace_impl(const T* A, linalg_int n, linalg_int lda, T* result, void* stream)
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

    LINALG_CHECK(result != nullptr, "device result must not be null");
    trace_kernel<T><<<1, 1, 0, stream_t>>>(A, detail::checked_narrow<int>(n), result);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "trace_kernel launch failed");
}

}  // namespace

template <>
void trace<float, linalg_backend::cuda_blas>::apply(
    const float* A, linalg_int n, linalg_int lda, float* result, void* stream)
{
    LINALG_SCOPE("trace/cuda_blas<float>");
    trace_impl(A, n, lda, result, stream);
}

template <>
void trace<double, linalg_backend::cuda_blas>::apply(
    const double* A, linalg_int n, linalg_int lda, double* result, void* stream)
{
    LINALG_SCOPE("trace/cuda_blas<double>");
    trace_impl(A, n, lda, result, stream);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
