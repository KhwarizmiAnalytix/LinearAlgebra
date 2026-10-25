#pragma once

#include <cstdint>
#include <limits>

#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"
#include "include/common/types.h"
#include "include/common/workspace_arena.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/schur_decomposition.h"

namespace linalg
{
namespace detail
{

inline int schur_function_order(linalg_int n)
{
    LINALG_CHECK(n > 0, "matrix function: n must be positive");
    const int order = checked_narrow<int>(n);
    LINALG_CHECK(order <= std::numeric_limits<int>::max() / order,
        "matrix function: matrix exceeds supported index range");
    return order;
}

// Shared scratch for every complex-Schur-based matrix function: the dedicated
// schur_decomposition<T, Backend> kernel supplies Q (qr/qi) and the triangular
// factor S (sr/si); this plan adds the real/imaginary planes each function
// reduces f(S) into, and the scratch reconstruct_schur_function() needs to
// fold Q*f(S)*Q^* back into a real result through existing backend GEMM.
template <typename T> struct schur_function_layout
{
    T*             sr;
    T*             si;
    T*             qr;
    T*             qi;
    T*             fr;
    T*             fi;
    T*             tr;
    T*             ti;
    T*             product1;
    T*             product2;
    T*             product3;
    T*             scratch_a;
    T*             scratch_b;
    unsigned char* schur_workspace;
    size_t         schur_workspace_bytes;
    unsigned char* gemm;
    size_t         gemm_bytes;
};

template <typename T, linalg_backend Backend>
schur_function_layout<T> schur_function_plan(
    workspace_arena& arena, linalg_int n, size_t alignment = 64)
{
    const auto               order = static_cast<size_t>(schur_function_order(n));
    const auto               nn    = order * order;
    schur_function_layout<T> p;
    p.sr                    = arena.next<T>(nn, alignment);
    p.si                    = arena.next<T>(nn, alignment);
    p.qr                    = arena.next<T>(nn, alignment);
    p.qi                    = arena.next<T>(nn, alignment);
    p.fr                    = arena.next<T>(nn, alignment);
    p.fi                    = arena.next<T>(nn, alignment);
    p.tr                    = arena.next<T>(nn, alignment);
    p.ti                    = arena.next<T>(nn, alignment);
    p.product1              = arena.next<T>(nn, alignment);
    p.product2              = arena.next<T>(nn, alignment);
    p.product3              = arena.next<T>(nn, alignment);
    p.scratch_a             = arena.next<T>(nn, alignment);
    p.scratch_b             = arena.next<T>(nn, alignment);
    p.schur_workspace_bytes = schur_decomposition<T, Backend>::workspace_size(n);
    p.schur_workspace       = arena.next<unsigned char>(p.schur_workspace_bytes, alignment);
    p.gemm_bytes            = multiplication<T, Backend>::workspace_size(n, n, n);
    p.gemm                  = arena.next<unsigned char>(p.gemm_bytes, alignment);
    return p;
}

// Q*F followed by (Q*F)*Q^*. Each complex product is folded into the
// Gauss/Karatsuba three-multiply identity (A*B: k1=ar*br, k2=ai*bi,
// k3=(ar+ai)*(br+bi); real=k1-k2, imag=k3-k1-k2), cutting every complex
// product from the naive four real GEMMs to three plus cheap O(n^2)
// elementwise combines, so native SIMD, BLAS, MKL and device backends keep
// their own GEMM kernels. combine(a, b, out, sign) runs on the same
// backend/stream as GEMM and tolerates out aliasing either input.
template <typename T, linalg_backend Backend, typename Combine>
void reconstruct_schur_function(
    const schur_function_layout<T>& p, linalg_int n, void* stream, Combine combine)
{
    const auto gemm = [&](const T* a, const T* b, T* out, bool transpose_b)
    {
        multiplication<T, Backend>::apply(
            false, transpose_b, n, n, n, a, n, b, n, out, n, p.gemm, p.gemm_bytes, stream);
    };

    // T = Q*F.
    combine(p.qr, p.qi, p.scratch_a, T(1));
    combine(p.fr, p.fi, p.scratch_b, T(1));
    gemm(p.qr, p.fr, p.product1, false);
    gemm(p.qi, p.fi, p.product2, false);
    gemm(p.scratch_a, p.scratch_b, p.product3, false);
    combine(p.product1, p.product2, p.tr, T(-1));
    combine(p.product3, p.product1, p.ti, T(-1));
    combine(p.ti, p.product2, p.ti, T(-1));

    // result = T*Q^*, with Q^* = qr^T - i*qi^T.
    combine(p.tr, p.ti, p.scratch_a, T(1));
    combine(p.qr, p.qi, p.scratch_b, T(-1));
    gemm(p.tr, p.qr, p.product1, true);
    gemm(p.ti, p.qi, p.product2, true);
    gemm(p.scratch_a, p.scratch_b, p.product3, true);
    combine(p.product1, p.product2, p.fr, T(1));
    combine(p.product3, p.product1, p.fi, T(-1));
    combine(p.fi, p.product2, p.fi, T(1));
}

template <typename T, linalg_backend Backend>
void check_schur_function_arguments(const T* a,
    linalg_int                               n,
    linalg_int                               lda,
    T*                                       result,
    linalg_int                               ldc,
    void*                                    workspace,
    size_t                                   workspace_size,
    size_t                                   total_bytes)
{
    const int order = schur_function_order(n);
    LINALG_CHECK(a != nullptr && result != nullptr, "matrix function: matrices must not be null");
    LINALG_CHECK(lda >= n && ldc >= n, "matrix function: row strides must be at least n");
    LINALG_CHECK(lda <= std::numeric_limits<int>::max() / order &&
                     ldc <= std::numeric_limits<int>::max() / order,
        "matrix function: row strides exceed supported index range");
    LINALG_CHECK(workspace != nullptr && workspace_size >= total_bytes,
        "matrix function: workspace is null or too small");
    LINALG_CHECK(reinterpret_cast<std::uintptr_t>(workspace) % (2 * alignof(T)) == 0,
        "matrix function: workspace must be aligned for its complex scalar type");
}

}  // namespace detail
}  // namespace linalg
