#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// `workspace` must point to at least `matrix_exponential_workspace_size(
// type_tag, n)` bytes of caller-owned host memory (the library never
// allocates its own memory — see include/common/workspace_arena.h). The
// query sizes the degree-13 Pade, which is the most scratch this routine
// uses, so the same buffer is valid for every matrix of order `n`.
LINALG_API size_t matrix_exponential_workspace_size(const float* /*type_tag*/, linalg_int n);
LINALG_API size_t matrix_exponential_workspace_size(const double* /*type_tag*/, linalg_int n);

namespace detail
{

// Host-memory-only CPU entry point. Computes result = exp(A) for a square,
// row-major n x n matrix A (lda/ldc >= n) via the scaling-and-squaring
// method with diagonal Pade approximants of degree 3/5/7/9/13, selected from
// ||A||_1 against the theta thresholds of N. J. Higham, "The Scaling and
// Squaring Method for the Matrix Exponential Revisited", SIAM J. Matrix
// Anal. Appl. 26(4), 2005 — the algorithm PyTorch's torch.matrix_exp /
// torch.linalg.matrix_exp and MATLAB's expm are built on. A is read-only;
// result may safely alias A.
//
// Composed entirely on top of linalg::detail::matmul_backend,
// linalg::detail::invert_backend, and linalg::detail::matrix_norm_backend
// (ONE) — there is no separate backend selection for this op itself, it
// simply inherits whichever CPU backend those were built with. Because of
// that, only the `native` specialization below is ever defined, and
// callers always name it explicitly rather than kActiveBackend. The
// template still exists, with the same explicit-member-specialization
// shape as cholesky_backend and friends, purely so this op's call site
// looks like every other op's. No GPU implementation exists for this op.
template <typename T, linalg_backend Backend>
struct matrix_exponential_backend
{
    LINALG_API static void apply(
        const T* A, linalg_int n, linalg_int lda, T* result, linalg_int ldc, void* workspace, size_t workspace_size);
};

}  // namespace detail

}  // namespace linalg
