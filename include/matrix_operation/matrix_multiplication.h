#pragma once

#include <cstddef>

#include "include/common/device.h"
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

// `device` tells the dispatcher whether a/b/c are host pointers or CUDA
// device pointers (see include/common/device.h); it cannot be inferred from
// the raw pointers themselves. Backend among CPU implementations (scalar /
// blas_lapack / mkl) is chosen automatically via linalg::globalContext(),
// overridable with linalg::globalContext().set_backend(...).
LINALG_API void matrix_multiplication(
    bool         transpose_a,
    bool         transpose_b,
    quarisma_int   rows,
    quarisma_int   columns,
    quarisma_int   depth,
    const float* a,
    quarisma_int   lda,
    const float* b,
    quarisma_int   ldb,
    float*       c,
    quarisma_int   ldc,
    device_type  device = device_type::cpu);

LINALG_API void matrix_multiplication(
    bool          transpose_a,
    bool          transpose_b,
    quarisma_int    rows,
    quarisma_int    columns,
    quarisma_int    depth,
    const double* a,
    quarisma_int    lda,
    const double* b,
    quarisma_int    ldb,
    double*       c,
    quarisma_int    ldc,
    device_type   device = device_type::cpu);
}  // namespace linalg