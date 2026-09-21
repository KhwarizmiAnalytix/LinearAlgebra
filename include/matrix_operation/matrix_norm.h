#pragma once

#include <cstddef>

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

enum class matrix_norm_type : linalg_int
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
LINALG_API float matrix_norm(
    const float* A, linalg_long rows, linalg_long columns, linalg_long lda, matrix_norm_type type);

LINALG_API double matrix_norm(
    const double* A, linalg_long rows, linalg_long columns, linalg_long lda, matrix_norm_type type);

}  // namespace linalg
