#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
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

// a/b/c are CUDA device pointers (cudaMalloc'd); only available when built
// with LINALG_ENABLE_CUBLAS. No host memory is read or written by this call.
// `stream` (default: the legacy default stream) is the CUDA stream the
// underlying cuBLAS call is issued on; this call does not synchronize it.
// For host pointers, see linalg::matrix_multiplication in
// include/matrix_operation/matrix_multiplication.h.
LINALG_API void matrix_multiplication(bool transpose_a,
    bool                                   transpose_b,
    quarisma_int                           rows,
    quarisma_int                           columns,
    quarisma_int                           depth,
    const float*                           a,
    quarisma_int                           lda,
    const float*                           b,
    quarisma_int                           ldb,
    float*                                 c,
    quarisma_int                           ldc,
    cudaStream_t                           stream = nullptr);

LINALG_API void matrix_multiplication(bool transpose_a,
    bool                                   transpose_b,
    quarisma_int                           rows,
    quarisma_int                           columns,
    quarisma_int                           depth,
    const double*                          a,
    quarisma_int                           lda,
    const double*                          b,
    quarisma_int                           ldb,
    double*                                c,
    quarisma_int                           ldc,
    cudaStream_t                           stream = nullptr);

}  // namespace gpu
}  // namespace linalg
