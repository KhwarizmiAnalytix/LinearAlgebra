#pragma once

#include <cstdint>

// LinearAlgebra — the CPU backend choices that matrix_operation/*.cpp
// specializes against (scalar fallback / BLAS+LAPACKE / MKL). There is no
// library-chosen default: every linalg::*<T, Backend> entry point requires
// the caller to name one of these explicitly, and LINALG_HAS_MKL,
// LINALG_HAS_BLAS, and LINALG_HAS_LAPACKE (always defined as 0 or 1 by
// CMakeLists.txt) only gate which specializations exist to be named.
namespace linalg
{

enum class linalg_backend : std::uint8_t
{
    native,
    blas_lapacke,
    mkl,
    cuda_blas
};

}  // namespace linalg
