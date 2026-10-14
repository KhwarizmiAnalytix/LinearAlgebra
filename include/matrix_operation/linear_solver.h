#pragma once

#include <cstddef>
#include <cstdint>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
enum class linear_solver_type : std::uint8_t
{
    LU_LINEAR_SOLVER               = 1,
    LU_UPFRONT_LINEAR_SOLVER       = 2,
    CHOLESKY_LINEAR_SOLVER         = 3,
    CHOLESKY_UPFRONT_LINEAR_SOLVER = 4,
};

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. For device (CUDA) pointers,
// see linalg::gpu::linear_solver in
// include/matrix_operation_gpu/linear_solver_gpu.h (no `pivot` parameter
// there — cuSOLVER manages LU pivots as internal device-side scratch).
LINALG_API void linear_solver(
    float*             m,
    linalg_int*        pivot,
    linalg_int         lda,
    float*             x,
    linear_solver_type type = linear_solver_type::LU_LINEAR_SOLVER);

LINALG_API void linear_solver(
    double*            m,
    linalg_int*        pivot,
    linalg_int         lda,
    double*            x,
    linear_solver_type type = linear_solver_type::LU_LINEAR_SOLVER);

}  // namespace linalg
