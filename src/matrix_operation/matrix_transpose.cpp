#include "include/matrix_operation/matrix_transpose.h"

#include <include/logging.h>
#include "include/common/narrow.h"

#include "include/common/workspace_arena.h"

#if LINALG_HAS_MKL
#include <mkl.h>
#elif LINALG_HAS_BLAS
#if LINALG_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <algorithm>
#else
#include <algorithm>
#endif

namespace linalg
{
namespace detail
{

// Null/positivity checks shared by every transpose_backend<T, Backend>::
// apply below, so callers (tests included) get the same validation
// regardless of which backend is active. There is no non-detail
// matrix_transpose() wrapper to put this in any more — callers use
// transpose_backend<T, kTransposeActiveBackend>::apply directly.
void check_transpose_inputs(const void* m, linalg_long rows, linalg_long columns)
{
    LOGGING_CHECK(m != nullptr, "transpose_backend::apply: m must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "transpose_backend::apply: rows and columns must be positive");
}

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that determines
// linalg::kTransposeActiveBackend (include/matrix_operation/
// matrix_transpose.h — this op's own backend axis, selected by
// LINALG_HAS_BLAS alone rather than LINALG_HAS_BLAS && LINALG_HAS_LAPACKE,
// since it only needs cblas_?copy). Each `apply` below is an explicit
// specialization of the one member matrix_transpose.h declares (not of the
// whole class), so it's defined exactly once, here, for whichever (T,
// Backend) pair is active — mkl.h / cblas.h stay out of the public header.
// Other translation units only ever see the header's declaration and link
// against this definition; they never instantiate a body of their own, so
// there's nothing to silently get wrong by omission.
#if LINALG_HAS_MKL

size_t transpose_mkl_workspace_size(linalg_long /*rows*/, linalg_long /*columns*/)
{
    return 0;
}

template <>
void transpose_backend<float, linalg_backend::mkl>::apply(
    linalg_long rows, linalg_long columns, float* m, void* /*workspace*/, size_t /*workspace_size*/)
{
    check_transpose_inputs(m, rows, columns);
    mkl_simatcopy('R', 'T', rows, columns, 1.F, m, columns, rows);
}

template <>
void transpose_backend<double, linalg_backend::mkl>::apply(
    linalg_long rows, linalg_long columns, double* m, void* /*workspace*/, size_t /*workspace_size*/)
{
    check_transpose_inputs(m, rows, columns);
    mkl_dimatcopy('R', 'T', rows, columns, 1., m, columns, rows);
}

#elif LINALG_HAS_BLAS

// No vendor-neutral CBLAS extension performs an in-place transpose (MKL's
// mkl_?imatcopy is an MKL-only extension), but a correct out-of-place one is
// just `rows` strided copies: row i of the row-major `rows x columns` source
// (contiguous, incx=1) becomes column i of the row-major `columns x rows`
// destination (stride `rows`, incy=rows). cblas_?copy is a real CBLAS entry
// point, so this runs on vendor-optimized copy kernels (OpenBLAS/Accelerate/
// netlib) rather than the scalar backend's cache-oblivious cycle-following
// in-place algorithm. `scratch` (rows*columns elements) is caller-owned,
// carved from `workspace`.
template <typename T> size_t transpose_blas_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    return arena.total_size();
}

size_t transpose_blas_workspace_size_f32(linalg_long rows, linalg_long columns)
{
    return transpose_blas_workspace_size_impl<float>(rows, columns);
}

size_t transpose_blas_workspace_size_f64(linalg_long rows, linalg_long columns)
{
    return transpose_blas_workspace_size_impl<double>(rows, columns);
}

template <>
void transpose_backend<float, linalg_backend::blas_lapacke>::apply(
    linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size)
{
    check_transpose_inputs(m, rows, columns);
    LOGGING_CHECK(workspace_size >= transpose_blas_workspace_size_f32(rows, columns),
        "matrix_transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   scratch = arena.next<float>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    // See the LP64-vs-ILP64 note on matmul_blas_f32 in matrix_multiplication.cpp:
    // macOS Accelerate deprecates these legacy LP64 entry points, but switching
    // to ACCELERATE_NEW_LAPACK is a separate, wider integer-width change.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    for (linalg_long i = 0; i < rows; ++i)
    {
        cblas_scopy(
            detail::checked_narrow<int>(columns), m + i * columns, 1, scratch + i, detail::checked_narrow<int>(rows));
    }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    std::copy(scratch, scratch + rows * columns, m);
}

template <>
void transpose_backend<double, linalg_backend::blas_lapacke>::apply(
    linalg_long rows, linalg_long columns, double* m, void* workspace, size_t workspace_size)
{
    check_transpose_inputs(m, rows, columns);
    LOGGING_CHECK(workspace_size >= transpose_blas_workspace_size_f64(rows, columns),
        "matrix_transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   scratch = arena.next<double>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    for (linalg_long i = 0; i < rows; ++i)
    {
        cblas_dcopy(
            detail::checked_narrow<int>(columns), m + i * columns, 1, scratch + i, detail::checked_narrow<int>(rows));
    }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    std::copy(scratch, scratch + rows * columns, m);
}

#else

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
void transpose_scalar_impl(linalg_long rows, linalg_long columns, T* m, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(workspace_size >= transpose_scalar_workspace_size_impl(rows, columns),
        "matrix_transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   visited =
        arena.next<unsigned char>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    transpose_cycles<T*>(m, m + rows * columns, static_cast<linalg_int>(columns), visited);
}

}  // namespace

size_t transpose_scalar_workspace_size_f32(linalg_long rows, linalg_long columns)
{
    return transpose_scalar_workspace_size_impl(rows, columns);
}

size_t transpose_scalar_workspace_size_f64(linalg_long rows, linalg_long columns)
{
    return transpose_scalar_workspace_size_impl(rows, columns);
}

template <>
void transpose_backend<float, linalg_backend::native>::apply(
    linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size)
{
    check_transpose_inputs(m, rows, columns);
    transpose_scalar_impl(rows, columns, m, workspace, workspace_size);
}

template <>
void transpose_backend<double, linalg_backend::native>::apply(
    linalg_long rows, linalg_long columns, double* m, void* workspace, size_t workspace_size)
{
    check_transpose_inputs(m, rows, columns);
    transpose_scalar_impl(rows, columns, m, workspace, workspace_size);
}

#endif

}  // namespace detail

size_t matrix_transpose_workspace_size(const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
#if LINALG_HAS_MKL
    return detail::transpose_mkl_workspace_size(rows, columns);
#elif LINALG_HAS_BLAS
    return detail::transpose_blas_workspace_size_f32(rows, columns);
#else
    return detail::transpose_scalar_workspace_size_f32(rows, columns);
#endif
}

size_t matrix_transpose_workspace_size(const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
#if LINALG_HAS_MKL
    return detail::transpose_mkl_workspace_size(rows, columns);
#elif LINALG_HAS_BLAS
    return detail::transpose_blas_workspace_size_f64(rows, columns);
#else
    return detail::transpose_scalar_workspace_size_f64(rows, columns);
#endif
}

}  // namespace linalg
