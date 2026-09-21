#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define linalg_int __int64
#define linalg_long unsigned __int64
#else
#define linalg_int long long int
#define linalg_long unsigned long long int
#endif

namespace linalg
{

// Host-memory-only CPU entry point for real symmetric matrices: A = V * D *
// V^T, with eigenvalues written to `eigenvalues` in ascending order (matches
// LAPACK syev(d)'s convention) and the corresponding orthonormal
// eigenvectors written as the columns of `eigenvectors` (n x n, row-major,
// ldv >= n). A is read-only; only its lower triangle is referenced. The CPU
// backend (scalar / blas_lapack / mkl) is a compile-time choice fixed by
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS (MKL takes precedence over BLAS
// over the scalar fallback) — there is no runtime dispatch or per-call
// backend override. No GPU implementation exists for this op. Returns false
// if the algorithm failed to converge.
LINALG_API bool symmetric_eigenvalue_decomposition(
    const float* A, linalg_int n, linalg_int lda, float* eigenvalues, float* eigenvectors, linalg_int ldv);

LINALG_API bool symmetric_eigenvalue_decomposition(
    const double* A, linalg_int n, linalg_int lda, double* eigenvalues, double* eigenvectors, linalg_int ldv);

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
LINALG_API bool eigenvalue_decomposition(
    const float* A,
    linalg_int n,
    linalg_int lda,
    float*       eigenvalues_real,
    float*       eigenvalues_imag,
    float*       eigenvectors,
    linalg_int ldv);

LINALG_API bool eigenvalue_decomposition(
    const double* A,
    linalg_int  n,
    linalg_int  lda,
    double*       eigenvalues_real,
    double*       eigenvalues_imag,
    double*       eigenvectors,
    linalg_int  ldv);

}  // namespace linalg
