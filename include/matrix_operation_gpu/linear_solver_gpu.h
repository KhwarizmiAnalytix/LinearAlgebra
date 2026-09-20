#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif

namespace linalg
{
namespace gpu
{

// `m`, `x` are CUDA device pointers; only available when built with
// LINALG_ENABLE_CUBLAS. Unlike the CPU overload, there is no `pivot`
// parameter: cuSOLVER manages LU pivots as its own internal device-side
// scratch (freed before this call returns). Success/failure is written into
// `info`, a device int* the caller owns — this call never copies to or from
// host memory, so it never blocks on a device synchronization. Inspect
// `info` explicitly (e.g. cudaMemcpyAsync on your own stream) only when/if
// you need the result.
LINALG_API void linear_solver(float* m, quarisma_int lda, float* x, linear_solver_type type, int* info);

LINALG_API void linear_solver(
    double* m, quarisma_int lda, double* x, linear_solver_type type, int* info);

}  // namespace gpu
}  // namespace linalg
