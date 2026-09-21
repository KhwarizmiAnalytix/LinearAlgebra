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

// Host-memory-only CPU entry point. Sum of the diagonal of a square,
// row-major n x n matrix. There is no vendor-library "trace" routine to
// call into (same rationale as matrix_determinant in matrix_inversion.cxx),
// so this always runs the same direct reduction regardless of
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS.
LINALG_API float matrix_trace(const float* A, linalg_int n, linalg_int lda);

LINALG_API double matrix_trace(const double* A, linalg_int n, linalg_int lda);

}  // namespace linalg
