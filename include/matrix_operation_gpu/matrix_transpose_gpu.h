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
namespace gpu
{

// `m` is a CUDA device pointer (cudaMalloc'd); only available when built
// with LINALG_ENABLE_CUBLAS. No host memory is read or written by this call.
// For host pointers, see linalg::matrix_transpose in
// include/matrix_operation/matrix_transpose.h.
LINALG_API void matrix_transpose(quarisma_long rows, quarisma_long columns, float* m);

LINALG_API void matrix_transpose(quarisma_long rows, quarisma_long columns, double* m);

}  // namespace gpu
}  // namespace linalg
