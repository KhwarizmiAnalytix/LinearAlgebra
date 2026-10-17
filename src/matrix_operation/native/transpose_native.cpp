#include "include/matrix_operation/transpose.h"

#include "include/common/linalg_checks.h"

#include "include/common/workspace_arena.h"

#include <algorithm>

#include "include/common/linalg_profiler.h"

namespace linalg
{

// Null/positivity checks shared by every transpose<T, Backend>::
// apply below, so callers (tests included) get the same validation
// regardless of which backend is active. There is no non-detail
// transpose() wrapper to put this in any more — callers use
// transpose<T, kTransposeActiveBackend>::apply directly.
static void check_transpose_inputs(const void* m, linalg_long rows, linalg_long columns)
{
    LINALG_CHECK(m != nullptr, "transpose::apply: m must not be null");
    LINALG_CHECK(rows != 0 && columns != 0, "transpose::apply: rows and columns must be positive");
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/matrix_transpose_mkl.cpp and
// blas/matrix_transpose_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

// `visited` is a caller-owned byte-per-element flag buffer (one byte, not a
// packed bit, for the simplest arena-compatible representation), carved
// from `workspace`; it is not initialized by the caller, so it must be
// zeroed here before use.
template <class RandomIterator>
void transpose_cycles(RandomIterator first, RandomIterator last, int m, unsigned char* visited)
{
    const int mn1 = (last - first - 1);
    const int n   = (last - first) / m;
    std::fill(visited, visited + (last - first), static_cast<unsigned char>(0));
    RandomIterator cycle = first;
    while (++cycle != last)
    {
        if (visited[cycle - first])
        {
            continue;
        }
        int a = cycle - first;
        do
        {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = 1;
        } while ((first + a) != cycle);
    }
}

size_t transpose_scalar_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    arena.next<unsigned char>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    return arena.total_size();
}

template <typename T>
void transpose_scalar_impl(
    linalg_long rows, linalg_long columns, T* m, void* workspace, size_t workspace_size)
{
    LINALG_CHECK(workspace_size >= transpose_scalar_workspace_size_impl(rows, columns),
        "transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   visited =
        arena.next<unsigned char>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    transpose_cycles<T*>(m, m + rows * columns, static_cast<linalg_int>(columns), visited);
}

}  // namespace

template <>
void transpose<float, linalg_backend::native>::apply(
    linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size)
{
    LINALG_SCOPE("transpose/native<float>");
    check_transpose_inputs(m, rows, columns);
    transpose_scalar_impl(rows, columns, m, workspace, workspace_size);
}

template <>
void transpose<double, linalg_backend::native>::apply(
    linalg_long rows, linalg_long columns, double* m, void* workspace, size_t workspace_size)
{
    LINALG_SCOPE("transpose/native<double>");
    check_transpose_inputs(m, rows, columns);
    transpose_scalar_impl(rows, columns, m, workspace, workspace_size);
}

template <>
size_t transpose<float, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_scalar_workspace_size_impl(rows, columns);
}

template <>
size_t transpose<double, linalg_backend::native>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_scalar_workspace_size_impl(rows, columns);
}
}  // namespace linalg
