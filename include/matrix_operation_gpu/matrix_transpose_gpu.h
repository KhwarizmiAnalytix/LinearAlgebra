#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
namespace gpu
{

// `m` is a CUDA device pointer (cudaMalloc'd); only available when built
// with LINALG_ENABLE_CUBLAS. No host memory is read or written by this call.
// `stream` (default: the legacy default stream) is the CUDA stream the
// underlying cuBLAS call and device-to-device copy are issued on; this call
// does not synchronize it. For host pointers, see linalg::matrix_transpose
// in include/matrix_operation/matrix_transpose.h.
LINALG_API void matrix_transpose(
    linalg_long rows, linalg_long columns, float* m, cudaStream_t stream = nullptr);

LINALG_API void matrix_transpose(
    linalg_long rows, linalg_long columns, double* m, cudaStream_t stream = nullptr);

}  // namespace gpu
}  // namespace linalg
