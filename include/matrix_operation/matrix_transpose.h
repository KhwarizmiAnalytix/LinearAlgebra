#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_HAS_MKL / LINALG_HAS_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. No vendor-neutral CBLAS
// extension performs an in-place transpose (MKL's mkl_?imatcopy is an
// MKL-only extension), so the blas_lapack branch instead does an
// out-of-place transpose via cblas_?copy strided row/column copies. For
// device (CUDA) pointers, see linalg::gpu::matrix_transpose in
// include/matrix_operation_gpu/matrix_transpose_gpu.h.
// `workspace` must point to at least `matrix_transpose_workspace_size(
// type_tag, rows, columns)` bytes of caller-owned host memory (the library
// never allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t matrix_transpose_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t matrix_transpose_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

LINALG_API void matrix_transpose(
    linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size);

LINALG_API void matrix_transpose(
    linalg_long rows, linalg_long columns, double* m, void* workspace, size_t workspace_size);

}  // namespace linalg
