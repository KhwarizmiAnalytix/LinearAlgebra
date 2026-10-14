#include "include/matrix_operation/matrix_transpose.h"

#include <include/logging.h>
#include "include/common/narrow.h"

#include "include/common/workspace_arena.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>
#elif defined(LINALG_ENABLE_BLAS)
#ifdef LINALG_BLAS_USE_ACCELERATE
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

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// matrix_transpose() overloads below call — exactly one is ever built.
#if defined(LINALG_ENABLE_MKL)

size_t transpose_mkl_workspace_size(linalg_long /*rows*/, linalg_long /*columns*/)
{
    return 0;
}

void transpose_mkl_f32(linalg_long rows, linalg_long columns, float* m, void* /*workspace*/, size_t /*workspace_size*/)
{
    mkl_simatcopy('R', 'T', rows, columns, 1.F, m, columns, rows);
}

void transpose_mkl_f64(linalg_long rows, linalg_long columns, double* m, void* /*workspace*/, size_t /*workspace_size*/)
{
    mkl_dimatcopy('R', 'T', rows, columns, 1., m, columns, rows);
}

#elif defined(LINALG_ENABLE_BLAS)

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

void transpose_blas_f32(linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(workspace_size >= transpose_blas_workspace_size_f32(rows, columns),
        "matrix_transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   scratch = arena.next<float>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    for (linalg_long i = 0; i < rows; ++i)
    {
        cblas_scopy(
            detail::checked_narrow<int>(columns), m + i * columns, 1, scratch + i, detail::checked_narrow<int>(rows));
    }
    std::copy(scratch, scratch + rows * columns, m);
}

void transpose_blas_f64(linalg_long rows, linalg_long columns, double* m, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(workspace_size >= transpose_blas_workspace_size_f64(rows, columns),
        "matrix_transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   scratch = arena.next<double>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    for (linalg_long i = 0; i < rows; ++i)
    {
        cblas_dcopy(
            detail::checked_narrow<int>(columns), m + i * columns, 1, scratch + i, detail::checked_narrow<int>(rows));
    }
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

void transpose_scalar_f32(linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size)
{
    transpose_scalar_impl(rows, columns, m, workspace, workspace_size);
}

void transpose_scalar_f64(linalg_long rows, linalg_long columns, double* m, void* workspace, size_t workspace_size)
{
    transpose_scalar_impl(rows, columns, m, workspace, workspace_size);
}

#endif

}  // namespace detail

size_t matrix_transpose_workspace_size(const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
#if defined(LINALG_ENABLE_MKL)
    return detail::transpose_mkl_workspace_size(rows, columns);
#elif defined(LINALG_ENABLE_BLAS)
    return detail::transpose_blas_workspace_size_f32(rows, columns);
#else
    return detail::transpose_scalar_workspace_size_f32(rows, columns);
#endif
}

size_t matrix_transpose_workspace_size(const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
#if defined(LINALG_ENABLE_MKL)
    return detail::transpose_mkl_workspace_size(rows, columns);
#elif defined(LINALG_ENABLE_BLAS)
    return detail::transpose_blas_workspace_size_f64(rows, columns);
#else
    return detail::transpose_scalar_workspace_size_f64(rows, columns);
#endif
}

void matrix_transpose(linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(m != nullptr, "matrix_transpose: m must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "matrix_transpose: rows and columns must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::transpose_mkl_f32(rows, columns, m, workspace, workspace_size);
#elif defined(LINALG_ENABLE_BLAS)
    detail::transpose_blas_f32(rows, columns, m, workspace, workspace_size);
#else
    detail::transpose_scalar_f32(rows, columns, m, workspace, workspace_size);
#endif
}

void matrix_transpose(linalg_long rows, linalg_long columns, double* m, void* workspace, size_t workspace_size)
{
    LOGGING_CHECK(m != nullptr, "matrix_transpose: m must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "matrix_transpose: rows and columns must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::transpose_mkl_f64(rows, columns, m, workspace, workspace_size);
#elif defined(LINALG_ENABLE_BLAS)
    detail::transpose_blas_f64(rows, columns, m, workspace, workspace_size);
#else
    detail::transpose_scalar_f64(rows, columns, m, workspace, workspace_size);
#endif
}

}  // namespace linalg
