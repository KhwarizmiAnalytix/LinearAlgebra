#include "include/matrix_operation/matrix_transpose.h"

#include "ThirdParty/Logging/include/logging.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>
#elif defined(LINALG_ENABLE_BLAS)
#ifdef LINALG_BLAS_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <algorithm>

#include "include/memory/allocator.h"
#else
#include <algorithm>
#include <vector>
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// matrix_transpose() overloads below call — exactly one is ever built.
#if defined(LINALG_ENABLE_MKL)

void transpose_mkl_f32(linalg_long rows, linalg_long columns, float* m)
{
    mkl_simatcopy('R', 'T', rows, columns, 1.F, m, columns, rows);
}

void transpose_mkl_f64(linalg_long rows, linalg_long columns, double* m)
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
// in-place algorithm.
void transpose_blas_f32(linalg_long rows, linalg_long columns, float* m)
{
    auto* scratch = allocator<float>::allocate(rows * columns);
    for (linalg_long i = 0; i < rows; ++i)
    {
        cblas_scopy(
            static_cast<int>(columns), m + i * columns, 1, scratch + i, static_cast<int>(rows));
    }
    std::copy(scratch, scratch + rows * columns, m);
    allocator<float>::free(scratch);
}

void transpose_blas_f64(linalg_long rows, linalg_long columns, double* m)
{
    auto* scratch = allocator<double>::allocate(rows * columns);
    for (linalg_long i = 0; i < rows; ++i)
    {
        cblas_dcopy(
            static_cast<int>(columns), m + i * columns, 1, scratch + i, static_cast<int>(rows));
    }
    std::copy(scratch, scratch + rows * columns, m);
    allocator<double>::free(scratch);
}

#else

namespace
{

template <class RandomIterator>
void transpose_cycles(RandomIterator first, RandomIterator last, int m)
{
    const int         mn1 = (last - first - 1);
    const int         n   = (last - first) / m;
    std::vector<bool> visited(last - first);
    RandomIterator    cycle = first;
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
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

template <typename T> void transpose_scalar_impl(linalg_long rows, linalg_long columns, T* m)
{
    transpose_cycles<T*>(m, m + rows * columns, static_cast<linalg_int>(columns));
}

}  // namespace

void transpose_scalar_f32(linalg_long rows, linalg_long columns, float* m)
{
    transpose_scalar_impl(rows, columns, m);
}

void transpose_scalar_f64(linalg_long rows, linalg_long columns, double* m)
{
    transpose_scalar_impl(rows, columns, m);
}

#endif

}  // namespace detail

void matrix_transpose(linalg_long rows, linalg_long columns, float* m)
{
    LOGGING_CHECK(m != nullptr, "matrix_transpose: m must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "matrix_transpose: rows and columns must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::transpose_mkl_f32(rows, columns, m);
#elif defined(LINALG_ENABLE_BLAS)
    detail::transpose_blas_f32(rows, columns, m);
#else
    detail::transpose_scalar_f32(rows, columns, m);
#endif
}

void matrix_transpose(linalg_long rows, linalg_long columns, double* m)
{
    LOGGING_CHECK(m != nullptr, "matrix_transpose: m must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "matrix_transpose: rows and columns must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::transpose_mkl_f64(rows, columns, m);
#elif defined(LINALG_ENABLE_BLAS)
    detail::transpose_blas_f64(rows, columns, m);
#else
    detail::transpose_scalar_f64(rows, columns, m);
#endif
}

}  // namespace linalg
