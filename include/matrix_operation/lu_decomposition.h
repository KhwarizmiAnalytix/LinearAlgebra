#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. For device (CUDA) pointers,
// see linalg::gpu::lu_decomposition in
// include/matrix_operation_gpu/lu_decomposition_gpu.h (same row-major
// "P * A = L * U" packed layout and pivot convention as this CPU entry
// point).
// `workspace` must point to at least `lu_decomposition_workspace_size(type_tag,
// lda)` bytes of caller-owned host memory (the library never allocates its
// own memory — see include/common/workspace_arena.h). The first argument to
// the size query is unused except to select the `float`/`double` overload
// you intend to call — pass `static_cast<const float*>(nullptr)` or the
// `double` equivalent.
LINALG_API size_t lu_decomposition_workspace_size(const float* /*type_tag*/, linalg_int lda);
LINALG_API size_t lu_decomposition_workspace_size(const double* /*type_tag*/, linalg_int lda);

LINALG_API bool lu_decomposition(
    float* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size);

LINALG_API bool lu_decomposition(
    double* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size);

}  // namespace linalg
