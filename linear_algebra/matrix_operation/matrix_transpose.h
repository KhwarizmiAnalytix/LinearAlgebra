#pragma once

#include <cstddef>

#include "linear_algebra/common/linear_algebra_export.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif

namespace linalg
{
LINALG_API void matrix_transpose(quarisma_long rows, quarisma_long columns, float* m);

LINALG_API void matrix_transpose(quarisma_long rows, quarisma_long columns, double* m);

}  // namespace linalg