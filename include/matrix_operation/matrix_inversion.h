#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"

namespace linalg
{

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapacke /
// mkl) is a compile-time choice fixed by linalg::kActiveBackend
// (include/common/backend.h) — MKL takes precedence over BLAS+LAPACKE over
// the scalar fallback, with no runtime dispatch or per-call override.
//
// `apply` and `workspace_size` are declared here but deliberately have NO
// body: this class template is never specialized as a whole, only these
// members are, via explicit member specializations defined once in
// matrix_inversion.cpp (for whichever backend LINALG_HAS_MKL /
// LINALG_HAS_BLAS && LINALG_HAS_LAPACKE selects — mkl.h / lapacke.h stay
// out of this public header). Every caller — including tests — sees only
// this declaration and calls
// `invert_backend<T, kActiveBackend>::apply(...)`; the actual function body
// is resolved by the linker against the one definition in
// matrix_inversion.cpp, so LINALG_API matters here (it must cross the
// shared-library boundary to reach the test binary) and there is no inline
// fallback body to silently instantiate per call site — calling this for a
// (T, Backend) nobody defined is a link error, not silently wrong behavior.
// For device (CUDA) pointers, see linalg::gpu::matrix_invert in
// include/matrix_operation_gpu/matrix_inversion_gpu.h.
template <typename T, linalg_backend Backend> struct invert_backend
{
    LINALG_API static void apply(T* m,
        linalg_int*                 pivot,
        linalg_int                  lda,
        linalg::linear_solver_type  type,
        void*                       workspace,
        size_t                      workspace_size);

    LINALG_API static size_t workspace_size(linalg_int lda, linalg::linear_solver_type type);
};

// Host-memory-only CPU entry point. Computes det(m) in-place (m is
// overwritten by the LU or Cholesky factorization). Built entirely on top of
// lu_backend / cholesky_backend (no separate vendor routine), so only the
// `native` specialization is ever defined and callers always name it
// explicitly rather than kActiveBackend. The template still exists with the
// same explicit-member-specialization shape as the other backends so this
// op's call site looks like every other op's.
template <typename T, linalg_backend Backend> struct matrix_determinant_backend
{
    LINALG_API static T apply(T*   m,
        linalg_int*                pivot,
        linalg_int                 lda,
        linalg::linear_solver_type type,
        void*                      workspace,
        size_t                     workspace_size);

    LINALG_API static size_t workspace_size(linalg_int lda, linalg::linear_solver_type type);
};

}  // namespace linalg
