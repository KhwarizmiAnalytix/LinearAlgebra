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
// `device` selects host vs. CUDA device pointers (see include/common/device.h);
// CPU backend (scalar / blas_lapack / mkl) is chosen via linalg::globalContext().
LINALG_API bool lu_decomposition(
    float* m, quarisma_int lda, quarisma_int* pivot, device_type device = device_type::cpu);

LINALG_API bool lu_decomposition(
    double* m, quarisma_int lda, quarisma_int* pivot, device_type device = device_type::cpu);

}  // namespace linalg