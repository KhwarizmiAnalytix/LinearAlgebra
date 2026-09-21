#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif

namespace linalg
{
namespace gpu
{

// `m` is a CUDA device pointer; only available when built with
// LINALG_ENABLE_CUBLAS. In place, exactly like the CPU overload: `m` holds
// A on input and A^-1 on output. Unlike the CPU overload there is no
// `pivot` parameter — cuSOLVER manages LU pivots as its own internal
// device-side scratch, matching linalg::gpu::linear_solver's convention
// (include/matrix_operation_gpu/linear_solver_gpu.h) rather than the CPU
// matrix_invert's caller-visible pivot array. Only LU_LINEAR_SOLVER and
// CHOLESKY_LINEAR_SOLVER are meaningful here: cuSOLVER always factors `m`
// itself on every call (there is no cuSOLVER equivalent of the CPU's
// "assume already factored" *_UPFRONT_LINEAR_SOLVER variants), so an
// *_UPFRONT_LINEAR_SOLVER type is treated identically to its non-upfront
// counterpart — the same simplification linalg::gpu::linear_solver already
// makes. Success/failure is written into `info`, a device int* the caller
// owns; this call never copies to or from host memory, so it never blocks
// on a device synchronization. `stream` (default: the legacy default
// stream) is the CUDA stream every underlying cuSOLVER/cuBLAS call and
// device-to-device copy is issued on.
LINALG_API void matrix_invert(float* m,
    quarisma_int                     lda,
    int*                             info,
    linear_solver_type               type   = linear_solver_type::LU_LINEAR_SOLVER,
    cudaStream_t                     stream = nullptr);

LINALG_API void matrix_invert(double* m,
    quarisma_int                      lda,
    int*                              info,
    linear_solver_type                type   = linear_solver_type::LU_LINEAR_SOLVER,
    cudaStream_t                      stream = nullptr);

// `m` is a CUDA device pointer; only available when built with
// LINALG_ENABLE_CUBLAS. `m` is factored internally exactly as matrix_invert
// above (same UPFRONT-collapsing note applies), then the determinant is
// computed via a small on-device reduction over the factor's diagonal (and,
// for the LU path, the pivot permutation's sign) — so this never downloads
// the full n x n factor to compute it — before being returned by value,
// matching the corresponding CPU matrix_determinant's signature exactly.
// Unlike every other linalg::gpu::* entry point, this does perform one
// internal device synchronization and a single-scalar device-to-host copy:
// returning by value means the result must actually be known on return,
// not deferred to a caller-side synchronization the way `info` is
// elsewhere in this API.
LINALG_API float matrix_determinant(float* m,
    quarisma_int                           lda,
    linear_solver_type                     type   = linear_solver_type::LU_LINEAR_SOLVER,
    cudaStream_t                           stream = nullptr);

LINALG_API double matrix_determinant(double* m,
    quarisma_int                             lda,
    linear_solver_type                       type   = linear_solver_type::LU_LINEAR_SOLVER,
    cudaStream_t                             stream = nullptr);

}  // namespace gpu
}  // namespace linalg
