#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// `workspace` must point to at least `qr_decomposition_workspace_size(
// type_tag, rows, columns)` bytes of caller-owned host memory (the library
// never allocates its own memory — see include/common/workspace_arena.h).
LINALG_API size_t qr_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t qr_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

namespace detail
{

// Host-memory-only CPU entry point. Economy ("thin") QR factorization of a
// general rows x columns, row-major matrix A: A = Q * R with
// k = min(rows, columns), Q sized rows x k with orthonormal columns
// (ldq >= k), and R sized k x columns, upper-trapezoidal (ldr >= columns;
// entries below the diagonal of R are written as zero). A is read-only. The
// CPU backend (scalar / blas_lapacke / mkl) is a compile-time choice fixed
// by linalg::kActiveBackend (include/common/backend.h) — MKL takes
// precedence over BLAS+LAPACKE over the scalar fallback, with no runtime
// dispatch or per-call override.
//
// `apply` is declared here but deliberately has NO body: this class template
// is never specialized as a whole, only this one member is, via an explicit
// member specialization defined once in qr_decomposition.cpp (for whichever
// backend LINALG_HAS_MKL / LINALG_HAS_BLAS && LINALG_HAS_LAPACKE selects —
// mkl.h / lapacke.h stay out of this public header). Every caller —
// including tests — sees only this declaration and calls
// `qr_backend<T, kActiveBackend>::apply(...)`; the actual function body is
// resolved by the linker against the one definition in qr_decomposition.cpp,
// so LINALG_API matters here (it must cross the shared-library boundary to
// reach the test binary) and there is no inline fallback body to silently
// instantiate per call site — calling this for a (T, Backend) nobody
// defined is a link error, not silently wrong behavior. For device (CUDA)
// pointers, see linalg::gpu::qr_decomposition in
// include/matrix_operation_gpu/qr_decomposition_gpu.h.
template <typename T, linalg_backend Backend>
struct qr_backend
{
    LINALG_API static void apply(
        linalg_long rows,
        linalg_long columns,
        const T*    A,
        linalg_long lda,
        T*          Q,
        linalg_long ldq,
        T*          R,
        linalg_long ldr,
        void*       workspace,
        size_t      workspace_size);
};

}  // namespace detail

}  // namespace linalg
