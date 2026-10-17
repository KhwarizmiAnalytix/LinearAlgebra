#if LINALG_HAS_BLAS

#include "include/matrix_operation/multiplication.h"

#include "include/common/narrow.h"
#include "include/common/linalg_checks.h"

#if LINALG_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include "include/common/linalg_profiler.h"

// BLAS specializations for multiplication.h (the native backend lives in
// multiplication.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
// on; the #if below is kept as a second guard.
namespace linalg
{

// Same checks as multiplication.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_matmul_inputs(const void* a,
    const void*                             b,
    const void*                             c,
    linalg_int                              rows,
    linalg_int                              columns,
    linalg_int                              depth,
    linalg_int                              lda,
    linalg_int                              ldb,
    linalg_int                              ldc)
{
    LINALG_CHECK(a != nullptr && b != nullptr && c != nullptr,
        "multiplication::apply: a, b, and c must not be null");
    LINALG_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0,
        "multiplication::apply: rows/columns/depth/lda/ldb/ldc must be positive");
}

template <>
void multiplication<float, linalg_backend::blas_lapacke>::apply(bool transpose_a,
    bool                                                             transpose_b,
    linalg_int                                                       rows,
    linalg_int                                                       columns,
    linalg_int                                                       depth,
    float const*                                                     a,
    linalg_int                                                       lda,
    float const*                                                     b,
    linalg_int                                                       ldb,
    float*                                                           c,
    linalg_int                                                       ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    LINALG_SCOPE("multiplication/blas_lapacke<float>");
    check_matmul_inputs(a, b, c, rows, columns, depth, lda, ldb, ldc);
    // macOS Accelerate deprecates the legacy LP64 cblas_* entry points in favor
    // of an ILP64 interface (ACCELERATE_NEW_LAPACK); switching would change the
    // integer width LAPACKE/checked_narrow assume throughout this codebase, so
    // we keep the legacy interface deliberately rather than as an oversight.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_sgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        detail::checked_narrow<int>(rows),
        detail::checked_narrow<int>(columns),
        detail::checked_narrow<int>(depth),
        1.F,
        a,
        detail::checked_narrow<int>(lda),
        b,
        detail::checked_narrow<int>(ldb),
        0.F,
        c,
        detail::checked_narrow<int>(ldc));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

template <>
void multiplication<double, linalg_backend::blas_lapacke>::apply(bool transpose_a,
    bool                                                              transpose_b,
    linalg_int                                                        rows,
    linalg_int                                                        columns,
    linalg_int                                                        depth,
    double const*                                                     a,
    linalg_int                                                        lda,
    double const*                                                     b,
    linalg_int                                                        ldb,
    double*                                                           c,
    linalg_int                                                        ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    LINALG_SCOPE("multiplication/blas_lapacke<double>");
    check_matmul_inputs(a, b, c, rows, columns, depth, lda, ldb, ldc);
    // See the LP64-vs-ILP64 note in matmul_blas_f32 above.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_dgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        detail::checked_narrow<int>(rows),
        detail::checked_narrow<int>(columns),
        detail::checked_narrow<int>(depth),
        1.,
        a,
        detail::checked_narrow<int>(lda),
        b,
        detail::checked_narrow<int>(ldb),
        0.,
        c,
        detail::checked_narrow<int>(ldc));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

template <>
size_t multiplication<float, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

template <>
size_t multiplication<double, linalg_backend::blas_lapacke>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

}  // namespace linalg
#endif  // LINALG_HAS_BLAS
