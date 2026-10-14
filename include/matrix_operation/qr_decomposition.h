#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// Host-memory-only CPU entry point. Economy ("thin") QR factorization of a
// general rows x columns, row-major matrix A: A = Q * R with
// k = min(rows, columns), Q sized rows x k with orthonormal columns
// (ldq >= k), and R sized k x columns, upper-trapezoidal (ldr >= columns;
// entries below the diagonal of R are written as zero). A is read-only. The
// CPU backend (scalar / blas_lapack / mkl) is a compile-time choice fixed by
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS (MKL takes precedence over BLAS
// over the scalar fallback) — there is no runtime dispatch or per-call
// backend override. For device (CUDA) pointers, see
// linalg::gpu::qr_decomposition in
// include/matrix_operation_gpu/qr_decomposition_gpu.h.
// `workspace` must point to at least `qr_decomposition_workspace_size(
// type_tag, rows, columns)` bytes of caller-owned host memory (the library
// never allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t qr_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t qr_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

LINALG_API void qr_decomposition(
    linalg_long rows,
    linalg_long columns,
    const float*  A,
    linalg_long lda,
    float*        Q,
    linalg_long ldq,
    float*        R,
    linalg_long ldr,
    void*         workspace,
    size_t        workspace_size);

LINALG_API void qr_decomposition(
    linalg_long rows,
    linalg_long columns,
    const double* A,
    linalg_long lda,
    double*       Q,
    linalg_long ldq,
    double*       R,
    linalg_long ldr,
    void*         workspace,
    size_t        workspace_size);

}  // namespace linalg
