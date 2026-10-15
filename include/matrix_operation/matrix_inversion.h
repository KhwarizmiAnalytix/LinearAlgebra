#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/matrix_operation/linear_solver.h"

namespace linalg
{

// `workspace` must point to at least `matrix_invert_workspace_size(type_tag,
// lda, type)` bytes of caller-owned host memory (the library never
// allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t matrix_invert_workspace_size(
    const float* /*type_tag*/, linalg_int lda, linalg::linear_solver_type type);
LINALG_API size_t matrix_invert_workspace_size(
    const double* /*type_tag*/, linalg_int lda, linalg::linear_solver_type type);

namespace detail
{

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapacke /
// mkl) is a compile-time choice fixed by linalg::kActiveBackend
// (include/common/backend.h) — MKL takes precedence over BLAS+LAPACKE over
// the scalar fallback, with no runtime dispatch or per-call override.
//
// `apply` is declared here but deliberately has NO body: this class template
// is never specialized as a whole, only this one member is, via an explicit
// member specialization defined once in matrix_inversion.cpp (for whichever
// backend LINALG_HAS_MKL / LINALG_HAS_BLAS && LINALG_HAS_LAPACKE selects —
// mkl.h / lapacke.h stay out of this public header). Every caller —
// including tests — sees only this declaration and calls
// `invert_backend<T, kActiveBackend>::apply(...)`; the actual function body
// is resolved by the linker against the one definition in
// matrix_inversion.cpp, so LINALG_API matters here (it must cross the
// shared-library boundary to reach the test binary) and there is no inline
// fallback body to silently instantiate per call site — calling this for a
// (T, Backend) nobody defined is a link error, not silently wrong behavior.
// For device (CUDA) pointers, see linalg::gpu::matrix_invert in
// include/matrix_operation_gpu/matrix_inversion_gpu.h.
template <typename T, linalg_backend Backend>
struct invert_backend
{
    LINALG_API static void apply(T* m,
        linalg_int*                pivot,
        linalg_int                 lda,
        linalg::linear_solver_type type,
        void*                      workspace,
        size_t                     workspace_size);
};

}  // namespace detail

// `workspace` must point to at least
// `matrix_determinant_workspace_size(type_tag, lda, type)` bytes.
LINALG_API size_t matrix_determinant_workspace_size(
    const float* /*type_tag*/, linalg_int lda, linalg::linear_solver_type type);
LINALG_API size_t matrix_determinant_workspace_size(
    const double* /*type_tag*/, linalg_int lda, linalg::linear_solver_type type);

LINALG_API float matrix_determinant(float* m,
    linalg_int*                           pivot,
    linalg_int                            lda,
    linalg::linear_solver_type            type,
    void*                                  workspace,
    size_t                                 workspace_size);

LINALG_API double matrix_determinant(double* m,
    linalg_int*                            pivot,
    linalg_int                             lda,
    linalg::linear_solver_type             type,
    void*                                   workspace,
    size_t                                  workspace_size);

}  // namespace linalg
