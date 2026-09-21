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

enum class cholesky_decomposition_enum : char
{
    LOWER_TRIANGULAR = 'L',
    UPPER_TRIANGULAR = 'U'
};

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapack /
// mkl) is a compile-time choice fixed by LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS
// (MKL takes precedence over BLAS over the scalar fallback) — there is no
// runtime dispatch or per-call backend override. For device (CUDA) pointers,
// see linalg::gpu::cholesky_decomposition in
// include/matrix_operation_gpu/cholesky_decomposition_gpu.h.
// cholesky_decomposition_aad (reverse-mode adjoint) has no vendor-library
// equivalent and always runs the scalar implementation.
LINALG_API bool cholesky_decomposition(
    float* L, linalg_int lda, linalg::cholesky_decomposition_enum type);

LINALG_API bool cholesky_decomposition_aad(
    float*                              L_aad,
    const float*                        L,
    linalg_int                          lda,
    linalg::cholesky_decomposition_enum type,
    float*                              A_aad);

LINALG_API bool cholesky_decomposition(
    double* L, linalg_int lda, linalg::cholesky_decomposition_enum type);

LINALG_API bool cholesky_decomposition_aad(
    double*                             L_aad,
    const double*                       L,
    linalg_int                          lda,
    linalg::cholesky_decomposition_enum type,
    double*                             A_aad);

}  // namespace linalg
