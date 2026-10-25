#pragma once

#include "src/matrix_operation/schur_function_workspace.h"

namespace linalg
{
namespace detail
{

// Adds the one complex n x n scratch square_root needs on top of the shared
// Schur-function plan: the triangular factor S, unpacked from sr/si into
// interleaved scalars::complex<T> storage for the triangular square-root recurrence.
template <typename T> struct square_root_layout
{
    schur_function_layout<T> schur;
    T*                       h;
    T*                       success;
};

template <typename T, linalg_backend Backend>
square_root_layout<T> square_root_plan(workspace_arena& arena, linalg_int n, size_t alignment = 64)
{
    square_root_layout<T> p;
    p.schur          = schur_function_plan<T, Backend>(arena, n, alignment);
    const auto order = static_cast<size_t>(schur_function_order(n));
    p.h              = arena.next<T>(2 * order * order, alignment);
    p.success        = arena.next<T>(1, alignment);
    return p;
}

template <typename T, linalg_backend Backend>
size_t square_root_workspace_size(linalg_int n, size_t alignment = 64)
{
    workspace_arena arena;
    square_root_plan<T, Backend>(arena, n, alignment);
    return arena.total_size();
}

}  // namespace detail
}  // namespace linalg
