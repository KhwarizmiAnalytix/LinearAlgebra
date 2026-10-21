#if LINALG_HAS_METAL

#include "include/matrix_operation/trace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

// Plain C++, not Objective-C++: trace only touches the n diagonal entries
// of an n x n matrix, so -- exactly like trace_cublas.cu/trace_hip.hip,
// which read the diagonal with a strided memcpy rather than launching a
// kernel for O(n) work -- there is nothing worth dispatching to the GPU
// here. METAL_BLAS's pointer is already host-visible (see
// include/common/metal_handle.h's file comment on MTLResourceStorageModeShared
// / unified memory), so this reads it directly with no Metal API calls at
// all; `stream` is accepted for API-signature parity with every other
// METAL_BLAS op but unused.
namespace linalg
{
namespace
{

float trace_impl(const float* A, linalg_int n, linalg_int lda)
{
    LINALG_CHECK(A != nullptr, "trace: A must not be null");
    LINALG_CHECK(n > 0, "trace: n must be positive");
    LINALG_CHECK(lda >= n, "trace: lda must be at least n");

    float sum = 0.0F;
    for (linalg_int i = 0; i < n; ++i)
    {
        sum += A[i * lda + i];
    }
    return sum;
}

}  // namespace

template <>
float trace<float, linalg_backend::METAL_BLAS>::apply(
    const float* A, linalg_int n, linalg_int lda, void* /*stream*/)
{
    LINALG_SCOPE("trace/metal_blas<float>");
    return trace_impl(A, n, lda);
}

}  // namespace linalg

#endif  // LINALG_HAS_METAL
