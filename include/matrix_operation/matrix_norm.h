#pragma once

#include <cstddef>
#include <cstdint>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

enum class matrix_norm_type : std::uint8_t
{
    FROBENIUS = 1,
    ONE       = 2,  // max absolute column sum
    INFINITY_NORM = 3,  // max absolute row sum
    TWO       = 4,  // spectral norm: largest singular value (via svd_decomposition)
};

// Host-memory-only CPU entry point. Works on general rows x columns,
// row-major matrices (not just square). FROBENIUS/ONE/INFINITY_NORM are a
// direct reduction over A and do not depend on LINALG_ENABLE_MKL /
// LINALG_ENABLE_BLAS. TWO routes through linalg::svd_decomposition, which is
// itself backend-dispatched (see svd_decomposition.h) — A is read-only here
// in every case.
// `workspace` must point to at least `matrix_norm_workspace_size(type_tag,
// rows, columns, type)` bytes of caller-owned host memory (the library
// never allocates its own memory) — the size depends on `type` since only
// TWO needs scratch at all (the workspace the internal svd_decomposition
// call needs; FROBENIUS/ONE/INFINITY_NORM need none).
LINALG_API size_t matrix_norm_workspace_size(const float* /*type_tag*/, linalg_long rows,
    linalg_long columns, matrix_norm_type type);
LINALG_API size_t matrix_norm_workspace_size(const double* /*type_tag*/, linalg_long rows,
    linalg_long columns, matrix_norm_type type);

LINALG_API float matrix_norm(const float* A, linalg_long rows, linalg_long columns, linalg_long lda,
    matrix_norm_type type, void* workspace, size_t workspace_size);

LINALG_API double matrix_norm(const double* A, linalg_long rows, linalg_long columns,
    linalg_long lda, matrix_norm_type type, void* workspace, size_t workspace_size);

}  // namespace linalg
