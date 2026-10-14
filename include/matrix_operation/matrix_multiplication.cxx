#include "include/matrix_operation/matrix_multiplication.h"

#include <include/logging.h>
#include "include/common/narrow.h"

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>
#elif defined(LINALG_ENABLE_BLAS)
#ifdef LINALG_BLAS_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#else
#if defined(_MSC_VER) && !defined(LINALG_DISPLAY_WIN32_WARNINGS)
#pragma warning(push)
#pragma warning(disable : 4244)
#endif  // _MSC_VER
#include <algorithm>
#include <cmath>
#include <limits>
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// matrix_multiplication() overloads below call — exactly one is ever built.
#if defined(LINALG_ENABLE_MKL)
void matmul_mkl_f32(bool transpose_a,
    bool                 transpose_b,
    linalg_int         rows,
    linalg_int         columns,
    linalg_int         depth,
    float const*         a,
    linalg_int         lda,
    float const*         b,
    linalg_int         ldb,
    float*               c,
    linalg_int         ldc)
{
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

void matmul_mkl_f64(bool transpose_a,
    bool                 transpose_b,
    linalg_int         rows,
    linalg_int         columns,
    linalg_int         depth,
    double const*        a,
    linalg_int         lda,
    double const*        b,
    linalg_int         ldb,
    double*              c,
    linalg_int         ldc)
{
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
#elif defined(LINALG_ENABLE_BLAS)
void matmul_blas_f32(bool transpose_a,
    bool                  transpose_b,
    linalg_int          rows,
    linalg_int          columns,
    linalg_int          depth,
    float const*          a,
    linalg_int          lda,
    float const*          b,
    linalg_int          ldb,
    float*                c,
    linalg_int          ldc)
{
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
}

void matmul_blas_f64(bool transpose_a,
    bool                  transpose_b,
    linalg_int          rows,
    linalg_int          columns,
    linalg_int          depth,
    double const*         a,
    linalg_int          lda,
    double const*         b,
    linalg_int          ldb,
    double*               c,
    linalg_int          ldc)
{
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
}
#else

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

namespace
{

template <typename value_t, bool transpose_a, bool transpose_b>
void matrix_multiplication_seq(  // NOLINT
    linalg_int   rows,
    linalg_int   columns,
    linalg_int   depth,
    value_t const* a,
    linalg_int   lda,
    value_t const* b,
    linalg_int   ldb,
    value_t*       c,
    linalg_int   ldc)
{
    using size_type = linalg_int;
    if constexpr (!transpose_a && !transpose_b)
    {
        for (size_type i = 0; i < rows; ++i)
        {
            auto* c_i = &C(i, 0);
            auto* a_i = &A(i, 0);
            std::fill_n(c_i, columns, static_cast<value_t>(0.));

            for (size_type k = 0; k < depth; ++k)
            {
                auto*      b_k      = &B(k, 0);
                const auto alpha_ik = a_i[k];
                for (size_type j = 0; j < columns; ++j)
                {
                    c_i[j] += alpha_ik * b_k[j];
                }
            }
        }
        return;
    }
    else if constexpr (transpose_a && !transpose_b)
    {
        std::fill_n(&C(0, 0), columns * rows, static_cast<value_t>(0.));
        for (size_type k = 0; k < depth; ++k)
        {
            auto* b_k = &B(k, 0);
            auto* a_k = &A(k, 0);
            for (size_type i = 0; i < rows; ++i)
            {
                const auto alpha_ki = a_k[i];
                auto*      c_i      = &C(i, 0);
                for (size_type j = 0; j < columns; ++j)
                {
                    c_i[j] += alpha_ki * b_k[j];
                }
            }
        }
        return;
    }
    else if constexpr (!transpose_a && transpose_b)
    {
        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < columns; ++j)
            {
                auto& c0                = C(i, j);
                c0                      = 0;
                const auto* const alpha = &A(i, 0);
                const auto* const beta  = &B(j, 0);
                for (size_type k = 0; k < depth; ++k)
                {
                    c0 += alpha[k] * beta[k];
                }
            }
        }
        return;
    }
    else
    {
        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < columns; ++j)
            {
                auto& c0 = C(i, j);
                c0       = 0;
                for (size_type k = 0; k < depth; ++k)
                {
                    c0 += A(k, i) * B(j, k);
                }
            }
        }
        return;
    }
}
}  // namespace
// C(i,j) =C[i*ldc + j] j belong to [0,columns] and i to [0,rows]
void matmul_scalar_f32(bool transpose_a,
    bool                    transpose_b,
    linalg_int            rows,
    linalg_int            columns,
    linalg_int            depth,
    float const*            a,
    linalg_int            lda,
    float const*            b,
    linalg_int            ldb,
    float*                  c,
    linalg_int            ldc)
{
    if (transpose_a && transpose_b)
    {
        matrix_multiplication_seq<float, true, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<float, true, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && transpose_b)
    {
        matrix_multiplication_seq<float, false, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<float, false, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
}

void matmul_scalar_f64(bool transpose_a,
    bool                    transpose_b,
    linalg_int            rows,
    linalg_int            columns,
    linalg_int            depth,
    double const*           a,
    linalg_int            lda,
    double const*           b,
    linalg_int            ldb,
    double*                 c,
    linalg_int            ldc)
{
    if (transpose_a && transpose_b)
    {
        matrix_multiplication_seq<double, true, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<double, true, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && transpose_b)
    {
        matrix_multiplication_seq<double, false, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<double, false, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
}

#undef A
#undef B
#undef C

#endif

}  // namespace detail

void matrix_multiplication(bool transpose_a,
    bool                        transpose_b,
    linalg_int                rows,
    linalg_int                columns,
    linalg_int                depth,
    float const*                a,
    linalg_int                lda,
    float const*                b,
    linalg_int                ldb,
    float*                      c,
    linalg_int                ldc)
{
    LOGGING_CHECK(a != nullptr && b != nullptr && c != nullptr, "matrix_multiplication: a, b, and c must not be null");
    LOGGING_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0, "matrix_multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::matmul_mkl_f32(transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#elif defined(LINALG_ENABLE_BLAS)
    detail::matmul_blas_f32(transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#else
    detail::matmul_scalar_f32(
        transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#endif
}

void matrix_multiplication(bool transpose_a,
    bool                        transpose_b,
    linalg_int                rows,
    linalg_int                columns,
    linalg_int                depth,
    double const*               a,
    linalg_int                lda,
    double const*               b,
    linalg_int                ldb,
    double*                     c,
    linalg_int                ldc)
{
    LOGGING_CHECK(a != nullptr && b != nullptr && c != nullptr, "matrix_multiplication: a, b, and c must not be null");
    LOGGING_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0, "matrix_multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::matmul_mkl_f64(transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#elif defined(LINALG_ENABLE_BLAS)
    detail::matmul_blas_f64(transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#else
    detail::matmul_scalar_f64(
        transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#endif
}

}  // namespace linalg
#if !defined(LINALG_ENABLE_MKL) && !defined(LINALG_ENABLE_BLAS)
#if defined(_MSC_VER) && !defined(LINALG_DISPLAY_WIN32_WARNINGS)
#pragma warning(pop)
#endif
#endif
