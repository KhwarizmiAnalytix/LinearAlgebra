#if LINALG_HAS_ROCBLAS

#include <hip/hip_runtime.h>
#include <vector>

#include "include/matrix_operation/trace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace
{

// `A` is row-major n x n, tightly packed (lda == n), so the diagonal entries
// sit at stride n+1; a single strided D2H copy reads them all directly,
// needing neither a kernel launch nor device scratch for the result.
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
        LINALG_THROW("trace<rocm_blas>: requires tightly packed lda (lda == n)");
    }
    const auto stream_t = static_cast<hipStream_t>(stream);
    const int  nn       = detail::checked_narrow<int>(n);

    std::vector<T> diagonal(static_cast<size_t>(nn));
    LINALG_CHECK(hipMemcpy2DAsync(diagonal.data(),
                     sizeof(T),
                     A,
                     sizeof(T) * (static_cast<size_t>(nn) + 1),
                     sizeof(T),
                     static_cast<size_t>(nn),
                     hipMemcpyDeviceToHost,
                     stream_t) == hipSuccess,
        "trace: diagonal copy failed");
    LINALG_CHECK(hipStreamSynchronize(stream_t) == hipSuccess, "trace: stream sync failed");

    T sum = T(0);
    for (int i = 0; i < nn; ++i)
    {
        sum += diagonal[static_cast<size_t>(i)];
    }
    return sum;
}

}  // namespace

template <>
float trace<float, linalg_backend::ROCM_BLAS>::apply(
    const float* A, linalg_int n, linalg_int lda, void* stream)
{
    LINALG_SCOPE("trace/rocm_blas<float>");
    return trace_impl(A, n, lda, stream);
}

template <>
double trace<double, linalg_backend::ROCM_BLAS>::apply(
    const double* A, linalg_int n, linalg_int lda, void* stream)
{
    LINALG_SCOPE("trace/rocm_blas<double>");
    return trace_impl(A, n, lda, stream);
}

}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
