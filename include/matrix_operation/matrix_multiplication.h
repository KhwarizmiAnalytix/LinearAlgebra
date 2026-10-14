#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// Host-memory-only CPU entry point. The CPU backend is a compile-time choice
// fixed by which of LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS was set at build
// time (MKL takes precedence over BLAS over the scalar fallback) — there is
// no runtime dispatch or per-call backend override. For device (CUDA)
// pointers, see linalg::gpu::matrix_multiplication in
// include/matrix_operation_gpu/matrix_multiplication_gpu.h.
// `workspace` must point to at least `matrix_multiplication_workspace_size(
// type_tag, rows, columns, depth)` bytes of caller-owned host memory (the
// library never allocates its own memory — see
// include/common/workspace_arena.h). Vendor backends need no scratch, so
// that query returns 0 and `workspace` may be null.
LINALG_API size_t matrix_multiplication_workspace_size(
    const float* /*type_tag*/, linalg_int rows, linalg_int columns, linalg_int depth);
LINALG_API size_t matrix_multiplication_workspace_size(
    const double* /*type_tag*/, linalg_int rows, linalg_int columns, linalg_int depth);

LINALG_API void matrix_multiplication(
    bool         transpose_a,
    bool         transpose_b,
    linalg_int   rows,
    linalg_int   columns,
    linalg_int   depth,
    const float* a,
    linalg_int   lda,
    const float* b,
    linalg_int   ldb,
    float*       c,
    linalg_int   ldc,
    void*        workspace,
    size_t       workspace_size);

LINALG_API void matrix_multiplication(
    bool          transpose_a,
    bool          transpose_b,
    linalg_int    rows,
    linalg_int    columns,
    linalg_int    depth,
    const double* a,
    linalg_int    lda,
    const double* b,
    linalg_int    ldb,
    double*       c,
    linalg_int    ldc,
    void*         workspace,
    size_t        workspace_size);
}  // namespace linalg
