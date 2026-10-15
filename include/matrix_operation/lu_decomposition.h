#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

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
// lu_decomposition.cpp (for whichever backend LINALG_HAS_MKL /
// LINALG_HAS_BLAS && LINALG_HAS_LAPACKE selects — mkl.h / lapacke.h stay
// out of this public header). Every caller — including linear_solver.cpp,
// matrix_inversion.cpp, and tests — sees only this declaration and calls
// `lu_backend<T, kActiveBackend>::apply(...)`; the actual function body is
// resolved by the linker against the one definition in lu_decomposition.cpp,
// so LINALG_API matters here (it must cross the shared-library boundary to
// reach the test binary) and there is no inline fallback body to silently
// instantiate per call site — calling this for a (T, Backend) nobody defined
// is a link error, not silently wrong behavior. For device (CUDA) pointers,
// see linalg::gpu::lu_decomposition in
// include/matrix_operation_gpu/lu_decomposition_gpu.h (same row-major
// "P * A = L * U" packed layout and pivot convention as this CPU entry
// point).
template <typename T, linalg_backend Backend> struct lu_backend
{
    LINALG_API static bool apply(
        T* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size);

    LINALG_API static size_t workspace_size(linalg_int lda);
};

}  // namespace linalg
