#pragma once

#include <cstddef>  // for linalg_long

#include "include/common/linear_algebra_export.h"  // for LINALG_API
#include "include/common/types.h"

namespace linalg
{
// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. For device (CUDA) pointers,
// see linalg::gpu::svd_decomposition in
// include/matrix_operation_gpu/svd_decomposition_gpu.h.
// `workspace` must point to at least `svd_decomposition_workspace_size(
// type_tag, rows, columns)` bytes of caller-owned host memory (the library
// never allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t svd_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t svd_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

LINALG_API void svd_decomposition(
    linalg_long rows,
    linalg_long columns,
    float*      A,
    linalg_long lda,
    float*      S,
    float*      U,
    linalg_long ldu,
    float*      VT,
    linalg_long ldv,
    void*       workspace,
    size_t      workspace_size);

LINALG_API void svd_decomposition(
    linalg_long rows,
    linalg_long columns,
    double*     A,
    linalg_long lda,
    double*     S,
    double*     U,
    linalg_long ldu,
    double*     VT,
    linalg_long ldv,
    void*       workspace,
    size_t      workspace_size);

}  // namespace linalg
