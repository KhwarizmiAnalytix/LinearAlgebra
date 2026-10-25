#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

/**
 * @brief Principal matrix square root of a real square matrix.
 *
 * This is a matrix function, not an elementwise square root or Cholesky factor.
 * The input must have no negative real eigenvalues. Zero eigenvalues are supported
 * when the triangular recurrence is defined, including positive semidefinite
 * diagonal matrices; a nilpotent Jordan block can have no square root.
 * T is float or double (Metal: float). Uses the Schur square-root algorithm
 * underlying TensorFlow's sqrtm, with complex triangular form internally.
 */
template <typename T, linalg_backend Backend> struct square_root
{
    /**
     * @brief Computes result * result = A to numerical precision.
     *
     * A and result are row-major n x n matrices, with row strides lda/ldc >= n.
     * The input is preserved unless result aliases A; exact aliasing is supported.
     * n must be positive. Workspace is caller-owned, aligned to at least 2 * alignof(T), and
     * must contain at least workspace_size(n) bytes and not overlap either matrix.
     * CPU backends use host pointers;
     * CUDA/HIP use device pointers for matrices and workspace. Metal uses page-aligned
     * shared buffers. No numerical buffers are allocated internally.
     *
     * CPU numerical/domain failure throws; CUDA/HIP stay asynchronous on stream and
     * write NaNs on numerical/domain failure. Metal waits for its command queue and
     * also writes NaNs on numerical/domain failure. Invalid host launch arguments
     * throw on all backends. A null stream selects the backend's default stream/queue.
     */
    LINALG_API static void apply(const T* A,
        linalg_int                        n,
        linalg_int                        lda,
        T*                                result,
        linalg_int                        ldc,
        void*                             workspace,
        size_t                            workspace_size,
        void*                             stream = nullptr);

    /** @brief Required bytes of host (CPU), device (CUDA/HIP), or shared (Metal) scratch. */
    LINALG_API static size_t workspace_size(linalg_int n);
};

}  // namespace linalg
