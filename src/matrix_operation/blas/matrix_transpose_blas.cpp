#if LINALG_HAS_BLAS

#include "include/matrix_operation/matrix_transpose.h"

#include "include/common/narrow.h"
#include <include/logging.h>

#include "include/common/workspace_arena.h"

#if LINALG_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <algorithm>

// BLAS specializations for matrix_transpose.h (the native backend lives in
// matrix_transpose.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
// on; the #if below is kept as a second guard.
namespace linalg
{

// Same checks as matrix_transpose.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_transpose_inputs(const void* m, linalg_long rows, linalg_long columns)
{
    LOGGING_CHECK(m != nullptr, "transpose_backend::apply: m must not be null");
    LOGGING_CHECK(
        rows != 0 && columns != 0, "transpose_backend::apply: rows and columns must be positive");
}

// No vendor-neutral CBLAS extension performs an in-place transpose (MKL's
// mkl_?imatcopy is an MKL-only extension), but a correct out-of-place one is
// just `rows` strided copies: row i of the row-major `rows x columns` source
// (contiguous, incx=1) becomes column i of the row-major `columns x rows`
// destination (stride `rows`, incy=rows). cblas_?copy is a real CBLAS entry
// point, so this runs on vendor-optimized copy kernels (OpenBLAS/Accelerate/
// netlib) rather than the scalar backend's cache-oblivious cycle-following
// in-place algorithm. `scratch` (rows*columns elements) is caller-owned,
// carved from `workspace`.
template <typename T>
size_t transpose_blas_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    return arena.total_size();
}

template <>
void transpose_backend<float, linalg_backend::blas_lapacke>::apply(
    linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size)
{
    check_transpose_inputs(m, rows, columns);
    LOGGING_CHECK(workspace_size >= transpose_blas_workspace_size_impl<float>(rows, columns),
        "matrix_transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto* scratch = arena.next<float>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    // See the LP64-vs-ILP64 note on matmul_blas_f32 in matrix_multiplication.cpp:
    // macOS Accelerate deprecates these legacy LP64 entry points, but switching
    // to ACCELERATE_NEW_LAPACK is a separate, wider integer-width change.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    for (linalg_long i = 0; i < rows; ++i)
    {
        cblas_scopy(detail::checked_narrow<int>(columns),
            m + i * columns,
            1,
            scratch + i,
            detail::checked_narrow<int>(rows));
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
    LOGGING_CHECK(workspace_size >= transpose_blas_workspace_size_impl<double>(rows, columns),
        "matrix_transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto* scratch = arena.next<double>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    for (linalg_long i = 0; i < rows; ++i)
    {
        cblas_dcopy(detail::checked_narrow<int>(columns),
            m + i * columns,
            1,
            scratch + i,
            detail::checked_narrow<int>(rows));
    }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    std::copy(scratch, scratch + rows * columns, m);
}

template <>
size_t transpose_backend<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_blas_workspace_size_impl<float>(rows, columns);
}

template <>
size_t transpose_backend<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_blas_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg
#endif  // LINALG_HAS_BLAS
