#pragma once

#include <cstddef>
#include <cstdint>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
enum class linear_solver_type : std::uint8_t
{
    LU_LINEAR_SOLVER               = 1,
    LU_UPFRONT_LINEAR_SOLVER       = 2,
    CHOLESKY_LINEAR_SOLVER         = 3,
    CHOLESKY_UPFRONT_LINEAR_SOLVER = 4,
};

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. For device (CUDA) pointers,
// see linalg::gpu::linear_solver in
// include/matrix_operation_gpu/linear_solver_gpu.h (no `pivot` parameter
// there — cuSOLVER manages LU pivots as internal device-side scratch).
// `workspace` must point to at least `linear_solver_workspace_size(type_tag,
// lda, type)` bytes of caller-owned host memory (the library never
// allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t linear_solver_workspace_size(
    const float* /*type_tag*/, linalg_int lda, linear_solver_type type);
LINALG_API size_t linear_solver_workspace_size(
    const double* /*type_tag*/, linalg_int lda, linear_solver_type type);

LINALG_API void linear_solver(float* m,
    linalg_int*                     pivot,
    linalg_int                      lda,
    float*                          x,
    linear_solver_type              type,
    void*                           workspace,
    size_t                          workspace_size);

LINALG_API void linear_solver(double* m,
    linalg_int*                      pivot,
    linalg_int                       lda,
    double*                          x,
    linear_solver_type               type,
    void*                            workspace,
    size_t                           workspace_size);

}  // namespace linalg
