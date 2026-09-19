#pragma once

namespace linalg
{

// Compute backend for a CPU-side matrix_operation call. `cublas` covers both
// cuBLAS (BLAS-level ops, e.g. GEMM) and cuSOLVER (factorizations); it is
// selected via device_type::cuda rather than living in this CPU-preference
// list — see include/common/context.h.
enum class backend
{
    scalar,       // portable fallback, always compiled in
    blas_lapack,  // vendor-neutral CBLAS/LAPACKE (OpenBLAS, Accelerate, netlib, ...)
    mkl,          // Intel MKL (CBLAS + LAPACKE surface)
    cublas        // NVIDIA cuBLAS / cuSOLVER, requires device_type::cuda pointers
};

}  // namespace linalg
