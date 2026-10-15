#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/norm.h"
#include "include/common/types.h"

namespace linalg
{
namespace gpu
{

// `A` is a CUDA device pointer; only available when built with
// LINALG_HAS_CUBLAS. Reuses linalg::norm_type
// (include/matrix_operation/norm.h) so callers switch backends
// without a second enum. `A` is read-only.
//
// FROBENIUS is a single cublasX?nrm2 over the buffer read as one flat
// vector (basis-independent, so no row-/column-major bookkeeping is
// needed). ONE/INFINITY_NORM reduce a column-sums-of-abs or
// row-sums-of-abs vector (one small kernel, size columns or rows) down to
// its largest entry via cublasI?amax. TWO routes through
// linalg::gpu::svd_decomposition for the largest singular value. Every
// variant performs one internal device synchronization and copies back a
// single scalar (like matrix_determinant_gpu) — never the full
// rows x columns matrix.
//
// RESTRICTION (matching svd_decomposition_gpu): `lda` must equal `columns`.
// `workspace` must point to at least `matrix_norm_workspace_size(type_tag,
// rows, columns, type)` bytes of caller-owned device memory (the library
// never allocates its own GPU memory) — the size depends on `type` since
// each norm variant needs different scratch (FROBENIUS needs none; TWO
// needs the most, including the workspace the internal svd_decomposition
// call needs).
LINALG_API size_t matrix_norm_workspace_size(const float* /*type_tag*/, linalg_long rows,
    linalg_long columns, norm_type type);
LINALG_API size_t matrix_norm_workspace_size(const double* /*type_tag*/, linalg_long rows,
    linalg_long columns, norm_type type);

LINALG_API float norm(const float* A,
    linalg_long                        rows,
    linalg_long                        columns,
    linalg_long                        lda,
    norm_type                     type,
    void*                                 workspace,
    size_t                                workspace_size,
    cudaStream_t                         stream = nullptr);

LINALG_API double norm(const double* A,
    linalg_long                         rows,
    linalg_long                         columns,
    linalg_long                         lda,
    norm_type                      type,
    void*                                  workspace,
    size_t                                 workspace_size,
    cudaStream_t                          stream = nullptr);

}  // namespace gpu
}  // namespace linalg
