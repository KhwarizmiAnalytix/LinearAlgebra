#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// `workspace` must point to at least `matrix_multiplication_workspace_size(
// type_tag, rows, columns, depth)` bytes of caller-owned host memory (the
// library never allocates its own memory — see
// include/common/workspace_arena.h). Vendor backends need no scratch, so
// that query returns 0 and `workspace` may be null.
LINALG_API size_t matrix_multiplication_workspace_size(
    const float* /*type_tag*/, linalg_int rows, linalg_int columns, linalg_int depth);
LINALG_API size_t matrix_multiplication_workspace_size(
    const double* /*type_tag*/, linalg_int rows, linalg_int columns, linalg_int depth);

// Unlike every other matrix_operation op, the BLAS branch here (and in
// matrix_transpose.h) only needs cblas_?gemm, not LAPACKE, so it is
// selected by LINALG_HAS_BLAS alone — LINALG_HAS_LAPACKE never factors in.
// That makes this op's own active-backend choice diverge from the global
// linalg::kActiveBackend (which requires BLAS *and* LAPACKE for
// blas_lapacke) whenever BLAS is enabled without LAPACKE: kActiveBackend
// would read `native` while this op actually built and runs its
// `blas_lapacke` specialization. kMatmulActiveBackend tracks this op's own
// #if condition so callers always name the specialization that was
// actually compiled.
inline constexpr linalg_backend kMatmulActiveBackend =
#if LINALG_HAS_MKL
    linalg_backend::mkl;
#elif LINALG_HAS_BLAS
    linalg_backend::blas_lapacke;
#else
    linalg_backend::native;
#endif

namespace detail
{

// Host-memory-only CPU entry point.
//
// `apply` is declared here but deliberately has NO body: this class template
// is never specialized as a whole, only this one member is, via an explicit
// member specialization defined once in matrix_multiplication.cpp (for
// whichever backend this op's own #if/#elif/#else selects — mkl.h / cblas.h
// stay out of this public header). Every caller — including tests — sees
// only this declaration and calls
// `matmul_backend<T, kMatmulActiveBackend>::apply(...)`; the actual
// function body is resolved by the linker against the one definition in
// matrix_multiplication.cpp, so LINALG_API matters here (it must cross the
// shared-library boundary to reach the test binary) and there is no inline
// fallback body to silently instantiate per call site — calling this for a
// (T, Backend) nobody defined is a link error, not silently wrong behavior.
// For device (CUDA) pointers, see linalg::gpu::matrix_multiplication in
// include/matrix_operation_gpu/matrix_multiplication_gpu.h.
template <typename T, linalg_backend Backend>
struct matmul_backend
{
    LINALG_API static void apply(
        bool       transpose_a,
        bool       transpose_b,
        linalg_int rows,
        linalg_int columns,
        linalg_int depth,
        const T*   a,
        linalg_int lda,
        const T*   b,
        linalg_int ldb,
        T*         c,
        linalg_int ldc,
        void*      workspace,
        size_t     workspace_size);
};

}  // namespace detail

}  // namespace linalg
