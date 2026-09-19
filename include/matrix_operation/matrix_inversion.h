#pragma once

#include <cstddef>

#include "include/common/device.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"

namespace linalg
{
// `device` selects host vs. CUDA device pointers (see include/common/device.h);
// CPU backend (scalar / blas_lapack / mkl) is chosen via linalg::globalContext().
// matrix_determinant has no device parameter: it only reads back the diagonal
// of an already-decomposed (host) buffer, which is not meaningful for device
// pointers without a host round trip the caller should do explicitly.
LINALG_API void matrix_invert(
    float*                     m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    linalg::linear_solver_type type   = linalg::linear_solver_type::LU_LINEAR_SOLVER,
    device_type                device = device_type::cpu);

LINALG_API void matrix_invert(
    double*                    m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    linalg::linear_solver_type type   = linalg::linear_solver_type::LU_LINEAR_SOLVER,
    device_type                device = device_type::cpu);

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