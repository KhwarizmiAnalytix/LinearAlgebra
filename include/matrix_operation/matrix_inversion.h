#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"

namespace linalg
{
// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. For device (CUDA) pointers,
// see linalg::gpu::matrix_invert in
// include/matrix_operation_gpu/matrix_inversion_gpu.h.
// `workspace` must point to at least `matrix_invert_workspace_size(type_tag,
// lda, type)` bytes of caller-owned host memory (the library never
// allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t matrix_invert_workspace_size(
    const float* /*type_tag*/, linalg_int lda, linalg::linear_solver_type type);
LINALG_API size_t matrix_invert_workspace_size(
    const double* /*type_tag*/, linalg_int lda, linalg::linear_solver_type type);

LINALG_API void matrix_invert(float* m,
    linalg_int*                     pivot,
    linalg_int                      lda,
    linalg::linear_solver_type      type,
    void*                           workspace,
    size_t                          workspace_size);

LINALG_API void matrix_invert(double* m,
    linalg_int*                      pivot,
    linalg_int                       lda,
    linalg::linear_solver_type       type,
    void*                            workspace,
    size_t                           workspace_size);

// `workspace` must point to at least
// `matrix_determinant_workspace_size(type_tag, lda, type)` bytes.
LINALG_API size_t matrix_determinant_workspace_size(
    const float* /*type_tag*/, linalg_int lda, linalg::linear_solver_type type);
LINALG_API size_t matrix_determinant_workspace_size(
    const double* /*type_tag*/, linalg_int lda, linalg::linear_solver_type type);

LINALG_API float matrix_determinant(float* m,
    linalg_int*                           pivot,
    linalg_int                            lda,
    linalg::linear_solver_type            type,
    void*                                  workspace,
    size_t                                 workspace_size);

LINALG_API double matrix_determinant(double* m,
    linalg_int*                            pivot,
    linalg_int                             lda,
    linalg::linear_solver_type             type,
    void*                                   workspace,
    size_t                                  workspace_size);

}  // namespace linalg
