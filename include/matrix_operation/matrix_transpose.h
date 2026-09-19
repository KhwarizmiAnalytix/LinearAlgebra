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
// `device` selects host vs. CUDA device pointers (see include/common/device.h).
// No vendor-neutral CBLAS extension exists for in-place non-square transpose
// (MKL's mkl_?imatcopy is an MKL-only extension), so the blas_lapack backend
// does not register for this op; it falls back to scalar under
// LINALG_ENABLE_BLAS. CPU backend (scalar / mkl) is otherwise chosen via
// linalg::globalContext().
LINALG_API void matrix_transpose(
    quarisma_long rows, quarisma_long columns, float* m, device_type device = device_type::cpu);

LINALG_API void matrix_transpose(
    quarisma_long rows, quarisma_long columns, double* m, device_type device = device_type::cpu);

}  // namespace linalg