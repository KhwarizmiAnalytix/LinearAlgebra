#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"

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

// `A`, `Ainv` are CUDA device pointers; only available when built with
// LINALG_ENABLE_CUBLAS. Moore-Penrose pseudo-inverse of a general
// rows x columns, row-major device matrix A, computed from its SVD
// (A+ = V * S+ * U^T, dropping singular values <= tol) via
// linalg::gpu::svd_decomposition — the same composition the CPU
// pseudo_inverse uses on top of the CPU svd_decomposition. `Ainv` is sized
// columns x rows (ldai >= rows), row-major. `A` is read-only.
//
// tol < 0 (the default) picks tol = eps(T) * max(rows, columns) * largest
// singular value; computing that requires knowing the largest singular
// value, so (like matrix_determinant_gpu) this call does perform one
// internal device synchronization and a single-scalar device-to-host copy
// (via cublasI?amax on the singular-value vector) before it can launch the
// combining kernel — every other buffer stays device-resident throughout.
//
// RESTRICTION (matching svd_decomposition_gpu / qr_decomposition_gpu):
// `lda` must equal `columns` and `ldai` must equal `rows` — every buffer
// must be tightly packed row-major with no padding.
LINALG_API void pseudo_inverse(linalg_long rows,
    linalg_long                            columns,
    const float*                             A,
    linalg_long                            lda,
    float*                                   Ainv,
    linalg_long                            ldai,
    float                                    tol    = -1.F,
    cudaStream_t                             stream = nullptr);

LINALG_API void pseudo_inverse(linalg_long rows,
    linalg_long                            columns,
    const double*                            A,
    linalg_long                            lda,
    double*                                  Ainv,
    linalg_long                            ldai,
    double                                   tol    = -1.,
    cudaStream_t                             stream = nullptr);

}  // namespace gpu
}  // namespace linalg
