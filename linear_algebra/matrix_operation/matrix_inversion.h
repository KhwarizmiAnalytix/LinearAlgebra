#pragma once

#include <cstddef>

#include "linear_algebra/common/linear_algebra_export.h"
#include "linear_algebra/matrix_operation/linear_solver.h"

namespace linalg
{
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