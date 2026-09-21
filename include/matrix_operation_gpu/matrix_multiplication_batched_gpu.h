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

// Batched multiplication of `count` independent `dim x dim` row-major
// matrices: C_i = A_i * B_i for i in [0, count). `a`, `b`, `c` are CUDA
// device pointers to `count` contiguous dim*dim-element matrices each
// (matrix i starts at a[i*dim*dim], etc.); only available when built with
// LINALG_ENABLE_CUBLAS. This runs a hand-written CUDA kernel (one thread per
// output element, one z-block per matrix) rather than cuBLAS, and reads or
// writes only device memory — no host copy is performed by this call, and
// it does not synchronize the stream it runs on; the caller is responsible
// for synchronizing before reading `c` on the host or elsewhere.
//
// This is a different operation from linalg::gpu::matrix_multiplication
// (single matrix, cuBLAS gemm, arbitrary shape/transpose): use this one only
// for many same-size square matrices at once. `stream` (default: the legacy
// default stream) is the CUDA stream the kernel is launched on.
LINALG_API void batched_matrix_multiplication(quarisma_int dim,
    quarisma_int                                           count,
    const float*                                           a,
    const float*                                           b,
    float*                                                 c,
    cudaStream_t                                           stream = nullptr);

LINALG_API void batched_matrix_multiplication(quarisma_int dim,
    quarisma_int                                           count,
    const double*                                          a,
    const double*                                          b,
    double*                                                c,
    cudaStream_t                                           stream = nullptr);

}  // namespace gpu
}  // namespace linalg
