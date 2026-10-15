#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

enum class cholesky_decomposition_enum : char
{
    LOWER_TRIANGULAR = 'L',
    UPPER_TRIANGULAR = 'U'
};

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapacke /
// mkl) is a compile-time choice fixed by linalg::kActiveBackend
// (include/common/backend.h) — MKL takes precedence over BLAS+LAPACKE over
// the scalar fallback, with no runtime dispatch or per-call override.
//
// `apply` is declared here but deliberately has NO body: this class template
// is never specialized as a whole, only these members are, via explicit
// member specializations defined once in cholesky_decomposition.cpp (for
// whichever backend LINALG_HAS_MKL / LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
// selects — mkl.h / lapacke.h stay out of this public header). Every caller
// — including linear_solver.cpp, matrix_inversion.cpp, and tests — sees only
// this declaration and calls `cholesky_backend<T, kActiveBackend>::apply(...)`;
// the actual function body is resolved by the linker against the one definition
// in cholesky_decomposition.cpp, so LINALG_API matters here (it must cross the
// shared-library boundary to reach the test binary) and there is no inline
// fallback body to silently instantiate per call site — calling this for a
// (T, Backend) nobody defined is a link error, not silently wrong behavior.
// `apply_aad` (reverse-mode adjoint) has no vendor-library equivalent and
// always runs the scalar implementation regardless of Backend. For device
// (CUDA) pointers, see linalg::gpu::cholesky_decomposition in
// include/matrix_operation_gpu/cholesky_decomposition_gpu.h.
template <typename T, linalg_backend Backend> struct cholesky_backend
{
    LINALG_API static bool apply(T* C, linalg_int lda, cholesky_decomposition_enum type);

    LINALG_API static bool apply_aad(
        T* L_aad, const T* L, linalg_int lda, cholesky_decomposition_enum type, T* A_aad);
};

}  // namespace linalg
