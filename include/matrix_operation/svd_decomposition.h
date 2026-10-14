#pragma once

#include <cstddef>  // for linalg_long

#include "include/common/linear_algebra_export.h"  // for LINALG_API
#include "include/common/types.h"

namespace linalg
{
// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. For device (CUDA) pointers,
// see linalg::gpu::svd_decomposition in
// include/matrix_operation_gpu/svd_decomposition_gpu.h.
LINALG_API void svd_decomposition(
    linalg_long rows,
    linalg_long columns,
    float*      A,
    linalg_long lda,
    float*      S,
    float*      U,
    linalg_long ldu,
    float*      VT,
    linalg_long ldv);

LINALG_API void svd_decomposition(
    linalg_long rows,
    linalg_long columns,
    double*     A,
    linalg_long lda,
    double*     S,
    double*     U,
    linalg_long ldu,
    double*     VT,
    linalg_long ldv);

}  // namespace linalg
