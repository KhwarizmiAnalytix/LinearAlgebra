#pragma once

#include <cstdint>
#include <limits>

#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"
#include "include/common/types.h"
#include "include/common/workspace_arena.h"

namespace linalg
{
namespace detail
{

inline int matrix_function_order(linalg_int n)
{
    LINALG_CHECK(n > 0, "matrix function: n must be positive");
    const int order = checked_narrow<int>(n);
    LINALG_CHECK(order <= std::numeric_limits<int>::max() / order,
        "matrix function: matrix exceeds supported index range");
    return order;
}

template <typename T> size_t matrix_function_workspace_size(linalg_int n)
{
    const auto order = static_cast<size_t>(matrix_function_order(n));
    // Five complex n*n matrices and one complex n-vector, stored as pairs of T.
    workspace_arena arena;
    arena.next<T>(10 * order * order + 2 * order);
    return arena.total_size();
}

template <typename T>
void check_matrix_function(const T* a,
    linalg_int                      n,
    linalg_int                      lda,
    T*                              result,
    linalg_int                      ldc,
    void*                           workspace,
    size_t                          workspace_size)
{
    const int order = matrix_function_order(n);
    LINALG_CHECK(a != nullptr && result != nullptr, "matrix function: matrices must not be null");
    LINALG_CHECK(lda >= n && ldc >= n, "matrix function: row strides must be at least n");
    LINALG_CHECK(lda <= std::numeric_limits<int>::max() / order &&
                     ldc <= std::numeric_limits<int>::max() / order,
        "matrix function: row strides exceed supported index range");
    LINALG_CHECK(workspace != nullptr && workspace_size >= matrix_function_workspace_size<T>(n),
        "matrix function: workspace is null or too small");
    LINALG_CHECK(reinterpret_cast<std::uintptr_t>(workspace) % alignof(T) == 0,
        "matrix function: workspace must be aligned for its scalar type");
}

}  // namespace detail
}  // namespace linalg
