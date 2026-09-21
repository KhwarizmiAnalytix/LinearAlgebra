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

// Host-memory-only CPU entry point. The CPU backend is a compile-time choice
// fixed by which of LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS was set at build
// time (MKL takes precedence over BLAS over the scalar fallback) — there is
// no runtime dispatch or per-call backend override. For device (CUDA)
// pointers, see linalg::gpu::matrix_multiplication in
// include/matrix_operation_gpu/matrix_multiplication_gpu.h.
LINALG_API void matrix_multiplication(
    bool         transpose_a,
    bool         transpose_b,
    linalg_int rows,
    linalg_int columns,
    linalg_int depth,
    const float* a,
    linalg_int lda,
    const float* b,
    linalg_int ldb,
    float*       c,
    linalg_int ldc);

LINALG_API void matrix_multiplication(
    bool          transpose_a,
    bool          transpose_b,
    linalg_int  rows,
    linalg_int  columns,
    linalg_int  depth,
    const double* a,
    linalg_int  lda,
    const double* b,
    linalg_int  ldb,
    double*       c,
    linalg_int  ldc);
}  // namespace linalg
