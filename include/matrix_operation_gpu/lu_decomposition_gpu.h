#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define linalg_int __int64
#define linalg_long unsigned __int64
#else
#define linalg_int long long int
#define linalg_long unsigned long long int
#endif

namespace linalg
{
namespace gpu
{

// `m`, `pivot`, `info` are CUDA device pointers; only available when built
// with LINALG_ENABLE_CUBLAS. `pivot` must point to at least `lda` ints of
// device memory and `info` to one; neither is copied to or from host memory
// by this call, so it never blocks on a device synchronization — inspect
// them explicitly (e.g. cudaMemcpyAsync on your own stream) only when/if you
// need the result. `stream` (default: the legacy default stream) is the
// CUDA stream the underlying cuSOLVER call is issued on.
//
// cuSOLVER is natively column-major; this entry point transposes `m` on the
// device before and after the factorization (see lu_decomposition_gpu.cxx)
// so the result matches the CPU lu_decomposition's row-major "P * A = L * U"
// packed layout and 1-based row-swap pivot convention exactly — a
// GPU-computed factor here is safe to compare against, or feed into
// reasoning about, the CPU-side convention. Both transposes stay entirely
// on the device (no host traffic), at the cost of two extra O(n^2)
// device-to-device copies per call.
LINALG_API void lu_decomposition(
    float* m, linalg_int lda, int* pivot, int* info, cudaStream_t stream = nullptr);

LINALG_API void lu_decomposition(
    double* m, linalg_int lda, int* pivot, int* info, cudaStream_t stream = nullptr);

}  // namespace gpu
}  // namespace linalg
