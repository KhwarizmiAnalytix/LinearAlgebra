#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// `workspace` must point to at least `lu_decomposition_workspace_size(type_tag,
// lda)` bytes of caller-owned host memory (the library never allocates its
// own memory — see include/common/workspace_arena.h). The first argument to
// the size query is unused except to select the `float`/`double` overload
// you intend to call — pass `static_cast<const float*>(nullptr)` or the
// `double` equivalent.
LINALG_API size_t lu_decomposition_workspace_size(const float* /*type_tag*/, linalg_int lda);
LINALG_API size_t lu_decomposition_workspace_size(const double* /*type_tag*/, linalg_int lda);

namespace detail
{

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapacke /
// mkl) is a compile-time choice fixed by linalg::kActiveBackend
// (include/common/backend.h) — MKL takes precedence over BLAS+LAPACKE over
// the scalar fallback, with no runtime dispatch or per-call override.
//
// `apply` is declared here but deliberately has NO body: this class template
// is never specialized as a whole, only this one member is, via an explicit
// member specialization defined once in lu_decomposition.cpp (for whichever
// backend LINALG_HAS_MKL / LINALG_HAS_BLAS && LINALG_HAS_LAPACKE selects —
// mkl.h / lapacke.h stay out of this public header). Every caller —
// including linear_solver.cpp, matrix_inversion.cpp, and tests — sees only
// this declaration and calls `lu_backend<T, kActiveBackend>::apply(...)`;
// the actual function body is resolved by the linker against the one
// definition in lu_decomposition.cpp, so LINALG_API matters here (it must
// cross the shared-library boundary to reach the test binary) and there is
// no inline fallback body to silently instantiate per call site — calling
// this for a (T, Backend) nobody defined is a link error, not silently
// wrong behavior. For device (CUDA) pointers, see linalg::gpu::lu_decomposition
// in include/matrix_operation_gpu/lu_decomposition_gpu.h (same row-major
// "P * A = L * U" packed layout and pivot convention as this CPU entry
// point).
template <typename T, linalg_backend Backend>
struct lu_backend
{
    LINALG_API static bool apply(
        T* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size);
};

}  // namespace detail

}  // namespace linalg
