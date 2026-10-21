#pragma once

#include <cstdint>

// LinearAlgebra — the backend choices that matrix_operation specializes
// against (scalar fallback / BLAS+LAPACKE / MKL / CUDA / ROCm). There is no
// library-chosen default: every linalg::*<T, Backend> entry point requires
// the caller to name one of these explicitly, and LINALG_HAS_MKL,
// LINALG_HAS_BLAS, LINALG_HAS_LAPACKE, LINALG_HAS_CUBLAS, and
// LINALG_HAS_ROCBLAS (defined as 0 or 1 by CMakeLists.txt) only gate which
// specializations exist to be named.
namespace linalg
{

enum class linalg_backend : std::uint8_t
{
    NATIVE,
    BLAS_LAPACKE,
    MKL,
    CUDA_BLAS,
    ROCM_BLAS
};

}  // namespace linalg
