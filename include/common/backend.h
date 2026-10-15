#pragma once

#include <cstdint>

// LinearAlgebra — single source of truth for the CPU backend choice that
// matrix_operation/*.cpp picks between (scalar fallback / BLAS+LAPACKE /
// MKL). LINALG_HAS_MKL, LINALG_HAS_BLAS, and LINALG_HAS_LAPACKE are always
// defined as 0 or 1 by CMakeLists.txt, so the precedence rule (MKL over
// BLAS+LAPACKE over scalar) can be computed once here instead of being
// re-derived with #if/#elif/#else at every call site.
namespace linalg
{

enum class linalg_backend : std::uint8_t
{
    native,
    blas_lapacke,
    mkl
};

inline constexpr linalg_backend kActiveBackend =
#if LINALG_HAS_MKL
    linalg_backend::mkl;
#elif LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    linalg_backend::blas_lapacke;
#else
    linalg_backend::native;
#endif

}  // namespace linalg
