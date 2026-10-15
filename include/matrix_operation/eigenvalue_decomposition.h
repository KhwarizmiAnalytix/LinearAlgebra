#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// `workspace` must point to at least
// `symmetric_eigenvalue_decomposition_workspace_size(type_tag, n)` bytes of
// caller-owned host memory (the library never allocates its own memory —
// see include/common/workspace_arena.h).
LINALG_API size_t symmetric_eigenvalue_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_int n);
LINALG_API size_t symmetric_eigenvalue_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_int n);

// `workspace` must point to at least
// `eigenvalue_decomposition_workspace_size(type_tag, n, compute_eigenvectors)`
// bytes. `compute_eigenvectors` must match whether the call passes a
// non-null `eigenvectors` pointer.
LINALG_API size_t eigenvalue_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_int n, bool compute_eigenvectors);
LINALG_API size_t eigenvalue_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_int n, bool compute_eigenvectors);

namespace detail
{

// Host-memory-only CPU entry point for real symmetric matrices: A = V * D *
// V^T, with eigenvalues written to `eigenvalues` in ascending order (matches
// LAPACK syev(d)'s convention) and the corresponding orthonormal
// eigenvectors written as the columns of `eigenvectors` (n x n, row-major,
// ldv >= n). A is read-only; only its lower triangle is referenced. The CPU
// backend (scalar / blas_lapacke / mkl) is a compile-time choice fixed by
// linalg::kActiveBackend (include/common/backend.h) — MKL takes precedence
// over BLAS+LAPACKE over the scalar fallback, with no runtime dispatch or
// per-call override. Returns false if the algorithm failed to converge. For
// device (CUDA) pointers, see linalg::gpu::symmetric_eigenvalue_decomposition
// in include/matrix_operation_gpu/eigenvalue_decomposition_gpu.h — the
// general (non-symmetric) backend below has no GPU counterpart.
//
// `apply` is declared here but deliberately has NO body: this class template
// is never specialized as a whole, only this one member is, via an explicit
// member specialization defined once in eigenvalue_decomposition.cpp (for
// whichever backend LINALG_HAS_MKL / LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
// selects — mkl.h / lapacke.h stay out of this public header). Every caller
// — including tests — sees only this declaration and calls
// `symmetric_eigen_backend<T, kActiveBackend>::apply(...)`; the actual
// function body is resolved by the linker against the one definition in
// eigenvalue_decomposition.cpp, so LINALG_API matters here (it must cross
// the shared-library boundary to reach the test binary) and there is no
// inline fallback body to silently instantiate per call site — calling this
// for a (T, Backend) nobody defined is a link error, not silently wrong
// behavior.
template <typename T, linalg_backend Backend>
struct symmetric_eigen_backend
{
    LINALG_API static bool apply(
        const T*   A,
        linalg_int n,
        linalg_int lda,
        T*         eigenvalues,
        T*         eigenvectors,
        linalg_int ldv,
        void*      workspace,
        size_t     workspace_size);
};

// Host-memory-only CPU entry point for general (non-symmetric) real square
// matrices. This library has no complex type (real-only, see repository
// conventions), so eigenvalues that come in complex-conjugate pairs are
// reported the way LAPACK's *geev does: `eigenvalues_real`/`eigenvalues_imag`
// hold the real/imaginary part of each of the n eigenvalues (in the order
// the algorithm converges them, not sorted), and for a complex-conjugate
// pair at positions j, j+1 (eigenvalues_imag[j] > 0,
// eigenvalues_imag[j+1] == -eigenvalues_imag[j]) the corresponding columns
// of `eigenvectors` hold the real part in column j and the imaginary part in
// column j+1 (so v_j = eigenvectors[:,j] + i*eigenvectors[:,j+1] and
// v_{j+1} = conj(v_j)). `eigenvectors` may be nullptr to skip eigenvector
// computation and only get eigenvalues.
//
// The scalar fallback computes eigenvalues for the general case via a
// real (shifted) QR algorithm and, when `eigenvectors` is non-null, real
// eigenvectors via inverse iteration on the original A; it throws if an
// eigenvector is requested for a genuinely complex eigenvalue (no complex
// arithmetic is available in the portable fallback — use the MKL/BLAS
// backend, whose LAPACKE *geev call packs complex eigenvectors natively, for
// that case). A is read-only. Returns false if the algorithm failed to
// converge.
//
// Declared/specialized the same way as symmetric_eigen_backend above —
// callers use `general_eigen_backend<T, kActiveBackend>::apply(...)`.
template <typename T, linalg_backend Backend>
struct general_eigen_backend
{
    LINALG_API static bool apply(
        const T*   A,
        linalg_int n,
        linalg_int lda,
        T*         eigenvalues_real,
        T*         eigenvalues_imag,
        T*         eigenvectors,
        linalg_int ldv,
        void*      workspace,
        size_t     workspace_size);
};

}  // namespace detail

}  // namespace linalg
