#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"

namespace linalg
{
// Host-memory-only CPU entry point (no GPU implementation exists for this
// op). The CPU backend (scalar / blas_lapack / mkl) is a compile-time choice
// fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS (MKL takes precedence over
// BLAS over the scalar fallback) — there is no runtime dispatch or per-call
// backend override.
LINALG_API void matrix_invert(
    float*                     m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    linalg::linear_solver_type type = linalg::linear_solver_type::LU_LINEAR_SOLVER);

LINALG_API void matrix_invert(
    double*                    m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    linalg::linear_solver_type type = linalg::linear_solver_type::LU_LINEAR_SOLVER);

LINALG_API float matrix_determinant(
    float*                     m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    linalg::linear_solver_type type = linalg::linear_solver_type::LU_LINEAR_SOLVER);

LINALG_API double matrix_determinant(
    double*                    m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    linalg::linear_solver_type type = linalg::linear_solver_type::LU_LINEAR_SOLVER);

}  // namespace linalg
