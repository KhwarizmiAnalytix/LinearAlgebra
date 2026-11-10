#if LINALG_HAS_MKL

#include "include/matrix_operation/multiplication.h"

#include <algorithm>

#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"

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

// alpha == 0 must make the product a no-op without reading a or b (see the BLAS
// contract documented on multiplication<T, Backend>::apply). Some MKL small-matrix
// kernels skip that quick-return internally and let 0 * NaN poison c, so this backend
// enforces the contract itself rather than trusting the vendor call for every size.
template <typename value_t>
static void scale_output(value_t* c, linalg_int rows, linalg_int columns, linalg_int ldc, value_t beta)
{
    if (beta == value_t(1))
    {
        return;
    }
    for (linalg_int i = 0; i < rows; ++i)
    {
        value_t* c_i = c + i * ldc;
        if (beta == value_t(0))
        {
            std::fill_n(c_i, columns, value_t(0));
        }
        else
        {
            for (linalg_int j = 0; j < columns; ++j)
            {
                c_i[j] *= beta;
            }
        }
    }
}

template <>
void multiplication<float, linalg_backend::MKL>::apply(bool transpose_a,
    bool                                                    transpose_b,
    linalg_int                                              rows,
    linalg_int                                              columns,
    linalg_int                                              depth,
    float                                                   alpha,
    float const*                                            a,
    linalg_int                                              lda,
    float const*                                            b,
    linalg_int                                              ldb,
    float                                                   beta,
    float*                                                  c,
    linalg_int                                              ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* /*stream*/)
{
    LINALG_SCOPE("multiplication/mkl<float>");
    check_matmul_inputs(a, b, c, rows, columns, depth, lda, ldb, ldc);
    if (alpha == 0.0f)
    {
        scale_output(c, rows, columns, ldc, beta);
        return;
    }
    cblas_sgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        detail::checked_narrow<MKL_INT>(rows),
        detail::checked_narrow<MKL_INT>(columns),
        detail::checked_narrow<MKL_INT>(depth),
        alpha,
        a,
        detail::checked_narrow<MKL_INT>(lda),
        b,
        detail::checked_narrow<MKL_INT>(ldb),
        beta,
        c,
        detail::checked_narrow<MKL_INT>(ldc));
}

template <>
void multiplication<double, linalg_backend::MKL>::apply(bool transpose_a,
    bool                                                     transpose_b,
    linalg_int                                               rows,
    linalg_int                                               columns,
    linalg_int                                               depth,
    double                                                   alpha,
    double const*                                            a,
    linalg_int                                               lda,
    double const*                                            b,
    linalg_int                                               ldb,
    double                                                   beta,
    double*                                                  c,
    linalg_int                                               ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* /*stream*/)
{
    LINALG_SCOPE("multiplication/mkl<double>");
    check_matmul_inputs(a, b, c, rows, columns, depth, lda, ldb, ldc);
    if (alpha == 0.0)
    {
        scale_output(c, rows, columns, ldc, beta);
        return;
    }
    cblas_dgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        detail::checked_narrow<MKL_INT>(rows),
        detail::checked_narrow<MKL_INT>(columns),
        detail::checked_narrow<MKL_INT>(depth),
        alpha,
        a,
        detail::checked_narrow<MKL_INT>(lda),
        b,
        detail::checked_narrow<MKL_INT>(ldb),
        beta,
        c,
        detail::checked_narrow<MKL_INT>(ldc));
}

template <>
size_t multiplication<float, linalg_backend::MKL>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

template <>
size_t multiplication<double, linalg_backend::MKL>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
