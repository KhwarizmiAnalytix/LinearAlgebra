#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

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
LINALG_API bool lu_decomposition(float* m, linalg_int lda, linalg_int* pivot);

LINALG_API bool lu_decomposition(double* m, linalg_int lda, linalg_int* pivot);

}  // namespace linalg
