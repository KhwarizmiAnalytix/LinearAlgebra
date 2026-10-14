#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/common/types.h"

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
// you need the result. `stream` (default: the legacy default stream) is the
// CUDA stream the underlying cuSOLVER call is issued on.
LINALG_API void linear_solver(float* m,
    linalg_int                     lda,
    float*                           x,
    linear_solver_type               type,
    int*                             info,
    cudaStream_t                     stream = nullptr);

LINALG_API void linear_solver(double* m,
    linalg_int                      lda,
    double*                           x,
    linear_solver_type                type,
    int*                              info,
    cudaStream_t                      stream = nullptr);

}  // namespace gpu
}  // namespace linalg
