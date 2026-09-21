#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
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
    linalg_int                           rows,
    linalg_int                           columns,
    linalg_int                           depth,
    const float*                           a,
    linalg_int                           lda,
    const float*                           b,
    linalg_int                           ldb,
    float*                                 c,
    linalg_int                           ldc,
    cudaStream_t                           stream = nullptr);

LINALG_API void matrix_multiplication(bool transpose_a,
    bool                                   transpose_b,
    linalg_int                           rows,
    linalg_int                           columns,
    linalg_int                           depth,
    const double*                          a,
    linalg_int                           lda,
    const double*                          b,
    linalg_int                           ldb,
    double*                                c,
    linalg_int                           ldc,
    cudaStream_t                           stream = nullptr);

}  // namespace gpu
}  // namespace linalg
