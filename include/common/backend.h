#pragma once

#include <cstdint>

// LinearAlgebra — the backend choices that matrix_operation specializes
// against (scalar fallback / BLAS+LAPACKE / MKL / CUDA / ROCm / Metal).
// There is no library-chosen default: every linalg::*<T, Backend> entry
// point requires the caller to name one of these explicitly, and
// LINALG_HAS_MKL, LINALG_HAS_BLAS, LINALG_HAS_LAPACKE, LINALG_HAS_CUBLAS,
// LINALG_HAS_ROCBLAS, and LINALG_HAS_METAL (defined as 0 or 1 by
// CMakeLists.txt) only gate which specializations exist to be named.
// METAL_BLAS is float-only (Apple GPUs have no native double type) and only
// covers the subset of ops Metal Performance Shaders genuinely accelerates
// (multiplication, batched multiplication, triangular_solver, transpose, norm, trace), plus
// a portable device schur_decomposition kernel that logarithm and square_root
// also build on — see src/matrix_operation/gpu/metal/ and
// docs/backend_roadmap.md §3.
namespace linalg
{

enum class linalg_backend : std::uint8_t
{
    NATIVE,
    BLAS_LAPACKE,
    MKL,
    CUDA_BLAS,
    ROCM_BLAS,
    METAL_BLAS
};

}  // namespace linalg
