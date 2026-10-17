#if LINALG_HAS_MKL

#include "include/matrix_operation/multiplication.h"

#include "include/common/linalg_checks.h"

#include <mkl.h>

#include "include/common/linalg_profiler.h"

// MKL specializations for multiplication.h (the native backend lives in
// multiplication.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
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
void multiplication<float, linalg_backend::mkl>::apply(bool transpose_a,
    bool                                                    transpose_b,
    linalg_int                                              rows,
    linalg_int                                              columns,
    linalg_int                                              depth,
    float const*                                            a,
    linalg_int                                              lda,
    float const*                                            b,
    linalg_int                                              ldb,
    float*                                                  c,
    linalg_int                                              ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    LINALG_SCOPE("multiplication/mkl<float>");
    check_matmul_inputs(a, b, c, rows, columns, depth, lda, ldb, ldc);
    cblas_sgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        rows,
        columns,
        depth,
        1.F,
        a,
        lda,
        b,
        ldb,
        0.F,
        c,
        ldc);
}

template <>
void multiplication<double, linalg_backend::mkl>::apply(bool transpose_a,
    bool                                                     transpose_b,
    linalg_int                                               rows,
    linalg_int                                               columns,
    linalg_int                                               depth,
    double const*                                            a,
    linalg_int                                               lda,
    double const*                                            b,
    linalg_int                                               ldb,
    double*                                                  c,
    linalg_int                                               ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    LINALG_SCOPE("multiplication/mkl<double>");
    check_matmul_inputs(a, b, c, rows, columns, depth, lda, ldb, ldc);
    cblas_dgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        rows,
        columns,
        depth,
        1.,
        a,
        lda,
        b,
        ldb,
        0.,
        c,
        ldc);
}

template <>
size_t multiplication<float, linalg_backend::mkl>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

template <>
size_t multiplication<double, linalg_backend::mkl>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
