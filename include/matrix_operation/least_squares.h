#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

// `workspace` must point to at least `least_squares_solve_workspace_size(
// type_tag, rows, columns)` bytes of caller-owned host memory (the library
// never allocates its own memory; this includes the workspace the internal
// pseudo_inverse call needs — see include/common/workspace_arena.h).
LINALG_API size_t least_squares_solve_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t least_squares_solve_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

namespace detail
{

// Host-memory-only CPU entry point. Solves the general linear least-squares
// problem min_X ||A*X - B||_2 for a rows x columns, row-major A and a
// rows x nrhs, row-major B: overdetermined (rows > columns) gives the
// least-squares solution, underdetermined (rows < columns) gives the
// minimum-norm solution, and rank-deficient A is handled the same way via
// the pseudo-inverse. X is sized columns x nrhs (ldx >= nrhs). A and B are
// read-only.
//
// Always routes through linalg::detail::pseudo_inverse_backend
// (X = A+ * B), which is itself backend-dispatched via
// linalg::detail::svd_backend — there is no separate backend selection for
// the reduction done here, so only the `native` specialization below is
// ever defined, and callers always name it explicitly rather than
// kActiveBackend. The template still exists, with the same
// explicit-member-specialization shape as cholesky_backend and friends,
// purely so this op's call site looks like every other op's. For device
// (CUDA) pointers, see linalg::gpu::least_squares_solve in
// include/matrix_operation_gpu/least_squares_gpu.h.
template <typename T, linalg_backend Backend>
struct least_squares_backend
{
    LINALG_API static void apply(
        linalg_long rows,
        linalg_long columns,
        linalg_long nrhs,
        const T*    A,
        linalg_long lda,
        const T*    B,
        linalg_long ldb,
        T*          X,
        linalg_long ldx,
        void*       workspace,
        size_t      workspace_size);
};

}  // namespace detail

}  // namespace linalg
