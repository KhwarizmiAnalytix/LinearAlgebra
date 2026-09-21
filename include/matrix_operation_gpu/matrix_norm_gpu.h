#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/matrix_norm.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define linalg_int __int64
#define linalg_long unsigned __int64
#else
#define linalg_int long long int
#define linalg_long unsigned long long int
#endif

namespace linalg
{
namespace gpu
{

// `A` is a CUDA device pointer; only available when built with
// LINALG_ENABLE_CUBLAS. Reuses linalg::matrix_norm_type
// (include/matrix_operation/matrix_norm.h) so callers switch backends
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
LINALG_API float matrix_norm(const float* A,
    linalg_long                        rows,
    linalg_long                        columns,
    linalg_long                        lda,
    matrix_norm_type                     type,
    cudaStream_t                         stream = nullptr);

LINALG_API double matrix_norm(const double* A,
    linalg_long                         rows,
    linalg_long                         columns,
    linalg_long                         lda,
    matrix_norm_type                      type,
    cudaStream_t                          stream = nullptr);

}  // namespace gpu
}  // namespace linalg
