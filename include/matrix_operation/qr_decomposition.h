#pragma once

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"
#include <cstddef>

namespace linalg
{
/** In-place packed QR (geqrf), followed optionally by Q formation (orgqr). */
template <typename T, linalg_backend Backend> struct qr_decomposition
{
    /**
     * Factor row-major rows x columns A in place (lda >= columns).
     * Writes k=min(rows,columns) Householder coefficients to caller-owned tau.
     * R occupies A's diagonal and upper triangle; entries below are reflectors,
     * not zeros. No separate R is extracted. The client decides whether to save R.
     * A and tau must remain unchanged until apply_q. Dimensions must be positive.
     * A, tau and workspace must not overlap. CUDA buffers must be device memory.
     */
    LINALG_API static void apply_r(linalg_long rows,
        linalg_long                            columns,
        T*                                     A,
        linalg_long                            lda,
        T*                                     tau,
        void*                                  workspace,
        size_t                                 workspace_size,
        void*                                  stream = nullptr);

    /**
     * Form the economy Q in the first k columns of A, using apply_r's packed A and tau.
     * Uses the original rows, columns and lda. Overwrites R and the reflectors;
     * there is no separate Q output or output copy. tau is read-only.
     * Scratch need not retain apply_r's contents and may use a different buffer.
     * On CUDA, order the calls on the same stream or synchronize between them.
     */
    LINALG_API static void apply_q(linalg_long rows,
        linalg_long                            columns,
        T*                                     A,
        linalg_long                            lda,
        const T*                               tau,
        void*                                  workspace,
        size_t                                 workspace_size,
        void*                                  stream = nullptr);

    /** Scratch bytes for packed factorization; excludes caller-owned A and tau. */
    LINALG_API static size_t workspace_size_r(linalg_long rows, linalg_long columns);
    /** Scratch bytes for Q formation; excludes caller-owned A and tau. */
    LINALG_API static size_t workspace_size_q(linalg_long rows, linalg_long columns);

    /** Scratch bytes for reusing a single buffer across both phases. */
    static size_t workspace_size(linalg_long rows, linalg_long columns)
    {
        const auto r = workspace_size_r(rows, columns);
        const auto q = workspace_size_q(rows, columns);
        return r > q ? r : q;
    }
};
}  // namespace linalg
