#pragma once

#include "include/matrix_operation/triangular_solver.h"
#include "src/matrix_operation/schur_function_workspace.h"

namespace linalg
{
namespace detail
{

inline void check_logarithm_order(linalg_int n)
{
    const int order = schur_function_order(n);
    LINALG_CHECK(order <= std::numeric_limits<int>::max() / order / 4,
        "logarithm: packed triangular system exceeds supported index range");
}

// Adds the complex scratch and the real packed-triangular-solve buffers the
// inverse-scaling/Gauss-Legendre logarithm needs on top of the shared
// Schur-function plan: h (the triangular factor S, unpacked into interleaved
// complex form), f (the Pade quadrature accumulator), one complex diagonal,
// three state scalars, and the seven-node triangular_solver<T, Backend> scratch.
template <typename T> struct logarithm_layout
{
    schur_function_layout<T> schur;
    T*                       h;
    T*                       f;
    T*                       diagonal;
    T*                       state;
    T*                       triangular_a;
    T*                       triangular_b;
    T*                       triangular_x;
    unsigned char*           triangular_workspace;
    size_t                   triangular_bytes;
};

template <typename T, linalg_backend Backend>
logarithm_layout<T> logarithm_plan(workspace_arena& arena, linalg_int n, size_t alignment = 64)
{
    const auto          order = static_cast<size_t>(schur_function_order(n));
    const auto          nn    = order * order;
    logarithm_layout<T> p;
    p.schur                = schur_function_plan<T, Backend>(arena, n, alignment);
    p.h                    = arena.next<T>(2 * nn, alignment);
    p.f                    = arena.next<T>(2 * nn, alignment);
    p.diagonal             = arena.next<T>(2 * order, alignment);
    p.state                = arena.next<T>(3, alignment);
    p.triangular_a         = arena.next<T>(4 * nn, alignment);
    p.triangular_b         = arena.next<T>(2 * nn, alignment);
    p.triangular_x         = arena.next<T>(2 * nn, alignment);
    p.triangular_bytes     = triangular_solver<T, Backend>::workspace_size(2 * n, n);
    p.triangular_workspace = arena.next<unsigned char>(p.triangular_bytes, alignment);
    return p;
}

template <typename T, linalg_backend Backend>
size_t logarithm_workspace_size(linalg_int n, size_t alignment = 64)
{
    workspace_arena arena;
    logarithm_plan<T, Backend>(arena, n, alignment);
    return arena.total_size();
}

}  // namespace detail
}  // namespace linalg
