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
// LINALG_HAS_CUBLAS. Unlike the CPU overload, there is no `pivot`
// parameter: cuSOLVER manages LU pivots as its own internal device-side
// scratch (freed before this call returns). Success/failure is written into
// `info`, a device int* the caller owns — this call never copies to or from
// host memory, so it never blocks on a device synchronization. Inspect
// `info` explicitly (e.g. cudaMemcpyAsync on your own stream) only when/if
// you need the result. `stream` (default: the legacy default stream) is the
// CUDA stream the underlying cuSOLVER call is issued on.
// `workspace` must point to at least
// `linear_solver_workspace_size(type_tag, lda, type)` bytes of caller-owned
// device memory (the library never allocates its own GPU memory); keep it
// alive and unmodified until work queued on `stream` has completed.
LINALG_API size_t linear_solver_workspace_size(
    const float* /*type_tag*/, linalg_int lda, linear_solver_type type);
LINALG_API size_t linear_solver_workspace_size(
    const double* /*type_tag*/, linalg_int lda, linear_solver_type type);

LINALG_API void linear_solver(float* m,
    linalg_int                     lda,
    float*                           x,
    linear_solver_type               type,
    int*                             info,
    void*                            workspace,
    size_t                           workspace_size,
    cudaStream_t                     stream = nullptr);

LINALG_API void linear_solver(double* m,
    linalg_int                      lda,
    double*                           x,
    linear_solver_type                type,
    int*                              info,
    void*                             workspace,
    size_t                            workspace_size,
    cudaStream_t                      stream = nullptr);

}  // namespace gpu
}  // namespace linalg
