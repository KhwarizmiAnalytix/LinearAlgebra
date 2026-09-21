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
// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. No vendor-neutral CBLAS
// extension performs an in-place transpose (MKL's mkl_?imatcopy is an
// MKL-only extension), so the blas_lapack branch instead does an
// out-of-place transpose via cblas_?copy strided row/column copies. For
// device (CUDA) pointers, see linalg::gpu::matrix_transpose in
// include/matrix_operation_gpu/matrix_transpose_gpu.h.
LINALG_API void matrix_transpose(linalg_long rows, linalg_long columns, float* m);

LINALG_API void matrix_transpose(linalg_long rows, linalg_long columns, double* m);

}  // namespace linalg
