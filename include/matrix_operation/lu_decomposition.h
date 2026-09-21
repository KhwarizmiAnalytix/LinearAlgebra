#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif

namespace linalg
{
// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. For device (CUDA) pointers,
// see linalg::gpu::lu_decomposition in
// include/matrix_operation_gpu/lu_decomposition_gpu.h (same row-major
// "P * A = L * U" packed layout and pivot convention as this CPU entry
// point).
LINALG_API bool lu_decomposition(float* m, quarisma_int lda, quarisma_int* pivot);

LINALG_API bool lu_decomposition(double* m, quarisma_int lda, quarisma_int* pivot);

}  // namespace linalg
