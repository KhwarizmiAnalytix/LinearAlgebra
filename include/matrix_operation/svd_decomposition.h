#pragma once

#include <cstddef>  // for linalg_long

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"  // for LINALG_API
#include "include/common/types.h"

namespace linalg
{

// `workspace` must point to at least `svd_decomposition_workspace_size(
// type_tag, rows, columns)` bytes of caller-owned host memory (the library
// never allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t svd_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t svd_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

namespace detail
{

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapacke /
// mkl) is a compile-time choice fixed by linalg::kActiveBackend
// (include/common/backend.h) — MKL takes precedence over BLAS+LAPACKE over
// the scalar fallback, with no runtime dispatch or per-call override.
//
// `apply` is declared here but deliberately has NO body: this class template
// is never specialized as a whole, only this one member is, via an explicit
// member specialization defined once in svd_decomposition.cpp (for whichever
// backend LINALG_HAS_MKL / LINALG_HAS_BLAS && LINALG_HAS_LAPACKE selects —
// mkl.h / lapacke.h stay out of this public header). Every caller —
// including tests — sees only this declaration and calls
// `svd_backend<T, kActiveBackend>::apply(...)`; the actual function body is
// resolved by the linker against the one definition in
// svd_decomposition.cpp, so LINALG_API matters here (it must cross the
// shared-library boundary to reach the test binary) and there is no inline
// fallback body to silently instantiate per call site — calling this for a
// (T, Backend) nobody defined is a link error, not silently wrong behavior.
// For device (CUDA) pointers, see linalg::gpu::svd_decomposition in
// include/matrix_operation_gpu/svd_decomposition_gpu.h.
template <typename T, linalg_backend Backend>
struct svd_backend
{
    LINALG_API static void apply(
        linalg_long rows,
        linalg_long columns,
        T*          A,
        linalg_long lda,
        T*          S,
        T*          U,
        linalg_long ldu,
        T*          VT,
        linalg_long ldv,
        void*       workspace,
        size_t      workspace_size);
};

}  // namespace detail

}  // namespace linalg
