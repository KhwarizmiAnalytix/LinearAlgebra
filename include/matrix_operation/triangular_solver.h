#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
/** Solves op(A) * X = B for a real triangular A and multiple right-hand sides.
 * Matrices are row-major: A is n x n, B and X are n x nrhs. Only the selected
 * triangle is read. Unit diagonal entries are implicit and are not read.
 * A, B, and X must occupy distinct buffers. A and B are preserved.
 * CPU uses host pointers; CUDA/HIP use device pointers and enqueue on stream;
 * Metal uses page-aligned shared buffers and waits for completion (float only).
 * Singular matrices are not diagnosed; callers must supply a nonsingular A.
 * All scratch is caller-owned and sized by workspace_size(n, nrhs).
 */
template <typename T, linalg_backend Backend> struct triangular_solver
{
    LINALG_API static void apply(bool upper,
        bool                          transpose,
        bool                          unit_diagonal,
        linalg_int                    n,
        linalg_int                    nrhs,
        const T*                      a,
        linalg_int                    lda,
        const T*                      b,
        linalg_int                    ldb,
        T*                            x,
        linalg_int                    ldx,
        void*                         workspace,
        size_t                        workspace_size,
        void*                         stream = nullptr);

    LINALG_API static size_t workspace_size(linalg_int n, linalg_int nrhs);
};
}  // namespace linalg
