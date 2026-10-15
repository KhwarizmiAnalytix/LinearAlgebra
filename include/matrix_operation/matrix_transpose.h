#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// `workspace` must point to at least `matrix_transpose_workspace_size(
// type_tag, rows, columns)` bytes of caller-owned host memory (the library
// never allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t matrix_transpose_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t matrix_transpose_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

// Unlike every other matrix_operation op, the BLAS branch here (and in
// matrix_multiplication.h) only needs cblas_?copy, not LAPACKE, so it is
// selected by LINALG_HAS_BLAS alone — LINALG_HAS_LAPACKE never factors in.
// That makes this op's own active-backend choice diverge from the global
// linalg::kActiveBackend (which requires BLAS *and* LAPACKE for
// blas_lapacke) whenever BLAS is enabled without LAPACKE: kActiveBackend
// would read `native` while this op actually built and runs its
// `blas_lapacke` specialization. kTransposeActiveBackend tracks this op's
// own #if condition so callers always name the specialization that was
// actually compiled.
inline constexpr linalg_backend kTransposeActiveBackend =
#if LINALG_HAS_MKL
    linalg_backend::mkl;
#elif LINALG_HAS_BLAS
    linalg_backend::blas_lapacke;
#else
    linalg_backend::native;
#endif

namespace detail
{

// Host-memory-only CPU entry point. No vendor-neutral CBLAS extension
// performs an in-place transpose (MKL's mkl_?imatcopy is an MKL-only
// extension), so the blas_lapacke branch instead does an out-of-place
// transpose via cblas_?copy strided row/column copies.
//
// `apply` is declared here but deliberately has NO body: this class template
// is never specialized as a whole, only this one member is, via an explicit
// member specialization defined once in matrix_transpose.cpp (for whichever
// backend this op's own #if/#elif/#else selects — mkl.h / cblas.h stay out
// of this public header). Every caller — including tests — sees only this
// declaration and calls `transpose_backend<T, kTransposeActiveBackend>::
// apply(...)`; the actual function body is resolved by the linker against
// the one definition in matrix_transpose.cpp, so LINALG_API matters here
// (it must cross the shared-library boundary to reach the test binary) and
// there is no inline fallback body to silently instantiate per call site —
// calling this for a (T, Backend) nobody defined is a link error, not
// silently wrong behavior. For device (CUDA) pointers, see
// linalg::gpu::matrix_transpose in
// include/matrix_operation_gpu/matrix_transpose_gpu.h.
template <typename T, linalg_backend Backend>
struct transpose_backend
{
    LINALG_API static void apply(
        linalg_long rows, linalg_long columns, T* m, void* workspace, size_t workspace_size);
};

}  // namespace detail

}  // namespace linalg
