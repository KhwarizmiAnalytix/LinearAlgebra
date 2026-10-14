#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
namespace gpu
{

// Number of bytes `lu_decomposition` needs in its `workspace` argument for a
// matrix of leading dimension `lda`; query this first and pass a
// caller-owned device buffer of at least this many bytes (the library never
// allocates its own GPU memory). The first argument is unused except to
// select the `float`/`double` overload matching the `lu_decomposition` call
// you intend to make — pass `static_cast<const float*>(nullptr)` or the
// `double` equivalent.
LINALG_API size_t lu_decomposition_workspace_size(const float* /*type_tag*/, linalg_int lda);
LINALG_API size_t lu_decomposition_workspace_size(const double* /*type_tag*/, linalg_int lda);

// `m`, `pivot`, `info`, `workspace` are CUDA device pointers; only available
// when built with LINALG_ENABLE_CUBLAS. `pivot` must point to at least `lda`
// ints of device memory, `info` to one, and `workspace` to at least
// `lu_decomposition_workspace_size<T>(lda)` bytes; none is copied to or from
// host memory by this call, so it never blocks on a device synchronization —
// inspect them explicitly (e.g. cudaMemcpyAsync on your own stream) only
// when/if you need the result. The caller must keep `workspace` alive and
// unmodified until work queued on `stream` by this call has completed.
// `stream` (default: the legacy default stream) is the CUDA stream the
// underlying cuSOLVER call is issued on.
//
// cuSOLVER is natively column-major; this entry point transposes `m` on the
// device before and after the factorization (see lu_decomposition_gpu.cpp)
// so the result matches the CPU lu_decomposition's row-major "P * A = L * U"
// packed layout and 1-based row-swap pivot convention exactly — a
// GPU-computed factor here is safe to compare against, or feed into
// reasoning about, the CPU-side convention. Both transposes stay entirely
// on the device (no host traffic), at the cost of two extra O(n^2)
// device-to-device copies per call.
LINALG_API void lu_decomposition(float* m, linalg_int lda, int* pivot, int* info,
    void* workspace, size_t workspace_size, cudaStream_t stream = nullptr);

LINALG_API void lu_decomposition(double* m, linalg_int lda, int* pivot, int* info,
    void* workspace, size_t workspace_size, cudaStream_t stream = nullptr);

}  // namespace gpu
}  // namespace linalg
