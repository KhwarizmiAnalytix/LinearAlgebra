#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
/** Complex Schur decomposition of a real square matrix: A = Q * S * Q^*.
 * S is complex upper triangular and Q is unitary. Real and imaginary parts
 * are returned in separate row-major n x n buffers with strides lds/ldq.
 * Input A (stride lda) is preserved; A, all four output planes, and workspace
 * must occupy distinct buffers. Output padding is preserved. Conjugate pairs,
 * repeated eigenvalues, defective matrices, and zero eigenvalues are supported.
 *
 * CPU supports float/double host buffers and throws on nonfinite input or
 * numerical failure. CUDA/HIP support float/double device buffers and enqueue
 * on stream without synchronizing; numerical failure writes NaNs to all four
 * planes. Metal supports float page-aligned shared buffers, waits for the
 * command queue, and also writes NaNs on numerical failure. Invalid host
 * arguments throw on every backend. All numerical scratch is caller-owned.
 */
template <typename T, linalg_backend Backend> struct schur_decomposition
{
    LINALG_API static void apply(const T* a,
        linalg_int                        n,
        linalg_int                        lda,
        T*                                schur_real,
        T*                                schur_imag,
        linalg_int                        lds,
        T*                                unitary_real,
        T*                                unitary_imag,
        linalg_int                        ldq,
        void*                             workspace,
        size_t                            workspace_size,
        void*                             stream = nullptr);

    LINALG_API static size_t workspace_size(linalg_int n);
};
}  // namespace linalg
