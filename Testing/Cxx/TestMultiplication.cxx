#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/multiplication.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

// #define DEBUG_MATRIXMULTIPLICATION

namespace
{
using linalg_test::dense_matrix;

template <typename value_t, linalg::linalg_backend Backend>
std::vector<unsigned char> gemm_workspace(linalg_int rows, linalg_int columns, linalg_int depth)
{
    const auto bytes =
        linalg::multiplication<value_t, Backend>::workspace_size(rows, columns, depth);
    return std::vector<unsigned char>(bytes);
}

template <typename> struct matrixTolerance
{
};

template <> struct matrixTolerance<float>
{
    static constexpr float tolerance = (float)1.5e-4;
};

template <> struct matrixTolerance<double>
{
    static constexpr double tolerance = 5.e-13;
};

// Fixed seed so a failure is reproducible. These values feed a tolerance
// check; a predictable sequence is intended.
std::default_random_engine test_generator()
{
    return std::default_random_engine{42};  // NOLINT(bugprone-random-generator-seed)
}

// v (1 x rows) * A (rows x columns) -> r (1 x columns), via the raw
// multiplication(transpose_a=false, transpose_b=false, ...) entry
// point, treating v as a 1-row matrix.
template <typename value_t, linalg::linalg_backend Backend>
void vector_matrix_multiplication(int rows, int columns)
{
    constexpr auto tol = matrixTolerance<value_t>::tolerance;

    dense_matrix<value_t> A(rows, columns);
    std::vector<value_t>  v(rows);
    std::vector<value_t>  r(columns);

    auto                                   generator = test_generator();
    std::uniform_real_distribution<double> distribution(-5., 5.);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                (value_t)distribution(generator);
        }
    }
    for (int j = 0; j < rows; ++j)
    {
        v[static_cast<std::size_t>(j)] = (value_t)distribution(generator);
    }

    auto gemm_ws = gemm_workspace<value_t, Backend>(1, columns, rows);
    linalg::multiplication<value_t, Backend>::apply(false,
        false,
        1,
        columns,
        rows,
        v.data(),
        rows,
        A.data(),
        columns,
        r.data(),
        columns,
        gemm_ws.data(),
        gemm_ws.size());

    std::vector<value_t> r_seq(static_cast<std::size_t>(columns));
    for (int j = 0; j < columns; ++j)
    {
        double sum = 0.;
        for (int i = 0; i < rows; ++i)
        {
            sum += v[static_cast<std::size_t>(i)] *
                   A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
        r_seq[static_cast<std::size_t>(j)] = (value_t)sum;
    }

    value_t max_error = 0;
    for (int j = 0; j < columns; ++j)
    {
        max_error = std::fmax(max_error,
            std::fabs(static_cast<double>(
                r[static_cast<std::size_t>(j)] - r_seq[static_cast<std::size_t>(j)])));
    }

#ifdef DEBUG_MATRIXMULTIPLICATION
    (void)max_error;
#else
    EXPECT_LT(max_error, tol);
#endif
}

// A (rows x columns) * v (columns x 1) -> r (rows x 1), plus a transpose
// sanity check (B = transpose(A) computed by hand, since transpose
// is not exercised by this test file — matches the original test scope).
template <typename value_t, linalg::linalg_backend Backend>
void matrix_vector_multiplication(int rows, int columns)
{
    constexpr auto tol = matrixTolerance<value_t>::tolerance;

    dense_matrix<value_t> A(rows, columns);
    std::vector<value_t>  v(columns);
    std::vector<value_t>  r(rows);

    auto                                   generator = test_generator();
    std::uniform_real_distribution<double> distribution(-5., 5.);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                (value_t)distribution(generator);
        }
    }
    for (int j = 0; j < columns; ++j)
    {
        v[static_cast<std::size_t>(j)] = (value_t)distribution(generator);
    }

    auto gemm_ws = gemm_workspace<value_t, Backend>(rows, 1, columns);
    linalg::multiplication<value_t, Backend>::apply(false,
        false,
        rows,
        1,
        columns,
        A.data(),
        columns,
        v.data(),
        1,
        r.data(),
        1,
        gemm_ws.data(),
        gemm_ws.size());

    double max_error = 0;
    for (int i = 0; i < rows; ++i)
    {
        double sum = 0.;
        for (int j = 0; j < columns; ++j)
        {
            sum += A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) *
                   v[static_cast<std::size_t>(j)];
        }
        max_error = std::fmax(max_error, std::fabs(sum - r[static_cast<std::size_t>(i)]));
    }

#ifdef DEBUG_MATRIXMULTIPLICATION
    (void)max_error;
#else
    EXPECT_LE(max_error, tol);
#endif

    dense_matrix<value_t> B(columns, rows);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            B(static_cast<std::size_t>(j), static_cast<std::size_t>(i)) =
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }

    max_error = 0.;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            max_error = std::fmax(max_error,
                std::fabs(static_cast<double>(
                    A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) -
                    B(static_cast<std::size_t>(j), static_cast<std::size_t>(i)))));
        }
    }
#ifdef DEBUG_MATRIXMULTIPLICATION
    (void)max_error;
#else
    EXPECT_LE(max_error, tol);
#endif
}

// General C(rows x columns) = alpha * op(A) * op(B) + beta * C, depth is the
// shared dimension. A's storage shape is (rows x depth) when !transpose_a,
// else (depth x rows); B's storage shape is (depth x columns) when
// !transpose_b, else (columns x depth) — matching linalg::multiplication's
// row-major convention (see include/matrix_operation/multiplication.h).
// alpha == 1 and beta == 0 go through the plain c = op(a) * op(b) overload so
// both entry points stay covered.
template <typename value_t, linalg::linalg_backend Backend>
void matrix_multiplication_test(int rows,
    int                             columns,
    int                             depth,
    bool                            transpose_a,
    bool                            transpose_b,
    int                             padding = 0,
    value_t                         alpha   = value_t(1),
    value_t                         beta    = value_t(0))
{
    SCOPED_TRACE(::testing::Message()
                 << "backend=" << static_cast<int>(Backend) << ", scalar bytes=" << sizeof(value_t)
                 << ", rows=" << rows << ", columns=" << columns << ", depth=" << depth
                 << ", transpose_a=" << transpose_a << ", transpose_b=" << transpose_b
                 << ", alpha=" << alpha << ", beta=" << beta);
    constexpr auto tol       = matrixTolerance<value_t>::tolerance;
    auto           a_rows    = rows;
    auto           a_columns = depth;
    if (transpose_a)
    {
        std::swap(a_rows, a_columns);
    }

    auto b_rows    = depth;
    auto b_columns = columns;
    if (transpose_b)
    {
        std::swap(b_rows, b_columns);
    }

    dense_matrix<value_t> A(a_rows, a_columns + padding);
    dense_matrix<value_t> B(b_rows, b_columns + padding);
    dense_matrix<value_t> C(rows, columns + padding);
    C.fill(value_t(123));

    auto                                   generator = test_generator();
    std::uniform_real_distribution<double> distribution(-5., 5.);

    for (int i = 0; i < a_rows; ++i)
    {
        for (int j = 0; j < a_columns; ++j)
        {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                (value_t)distribution(generator);
        }
    }

    for (int i = 0; i < b_rows; ++i)
    {
        for (int j = 0; j < b_columns; ++j)
        {
            B(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                (value_t)distribution(generator);
        }
    }

    // With beta != 0 the prior contents of C take part in the result; keep a
    // copy for the reference. With beta == 0 C keeps its 123 fill, which must
    // not leak into the result.
    if (beta != value_t(0))
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < columns; ++j)
            {
                C(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    (value_t)distribution(generator);
            }
        }
    }
    const dense_matrix<value_t> C0 = C;

    auto gemm_ws = gemm_workspace<value_t, Backend>(rows, columns, depth);
    if (alpha == value_t(1) && beta == value_t(0))
    {
        linalg::multiplication<value_t, Backend>::apply(transpose_a,
            transpose_b,
            rows,
            columns,
            depth,
            A.data(),
            static_cast<linalg_int>(A.cols()),
            B.data(),
            static_cast<linalg_int>(B.cols()),
            C.data(),
            static_cast<linalg_int>(C.cols()),
            gemm_ws.data(),
            gemm_ws.size());
    }
    else
    {
        linalg::multiplication<value_t, Backend>::apply(transpose_a,
            transpose_b,
            rows,
            columns,
            depth,
            alpha,
            A.data(),
            static_cast<linalg_int>(A.cols()),
            B.data(),
            static_cast<linalg_int>(B.cols()),
            beta,
            C.data(),
            static_cast<linalg_int>(C.cols()),
            gemm_ws.data(),
            gemm_ws.size());
    }

    // A dot product's rounding error depends on its depth and the magnitudes
    // of its products, including when cancellation leaves a small result.
    // Use a wider reference and the standard gamma_(2*depth) forward-error
    // bound instead of treating a sequential value_t sum as exact. The
    // existing absolute tolerance remains a floor for the small cases. Four
    // extra roundings cover the alpha scaling, the beta scaling, and the
    // final addition.
    const long double roundoff = std::numeric_limits<value_t>::epsilon() / 2.0L;
    const long double steps    = 2.0L * depth + 4.0L;
    const long double gamma    = (steps * roundoff) / (1.0L - steps * roundoff);
    long double       max_error_ratio = 0;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            long double sum          = 0;
            long double absolute_sum = 0;
            for (int k = 0; k < depth; ++k)
            {
                const value_t a_val =
                    transpose_a ? A(static_cast<std::size_t>(k), static_cast<std::size_t>(i))
                                : A(static_cast<std::size_t>(i), static_cast<std::size_t>(k));
                const value_t b_val =
                    transpose_b ? B(static_cast<std::size_t>(j), static_cast<std::size_t>(k))
                                : B(static_cast<std::size_t>(k), static_cast<std::size_t>(j));
                const long double product =
                    static_cast<long double>(a_val) * static_cast<long double>(b_val);
                sum += product;
                absolute_sum += std::fabs(product);
            }
            const long double prior = beta == value_t(0)
                                          ? 0.0L
                                          : static_cast<long double>(beta) *
                                                static_cast<long double>(C0(
                                                    static_cast<std::size_t>(i),
                                                    static_cast<std::size_t>(j)));
            const long double expected = static_cast<long double>(alpha) * sum + prior;
            const long double magnitude =
                std::fabs(static_cast<long double>(alpha)) * absolute_sum + std::fabs(prior);

            const auto c = C(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            ASSERT_TRUE(std::isfinite(c)) << "row=" << i << ", column=" << j;
            const long double allowed_error =
                std::max(static_cast<long double>(tol), gamma * magnitude);
            max_error_ratio = std::max(max_error_ratio,
                std::fabs(expected - static_cast<long double>(c)) / allowed_error);
        }
    }

    for (int i = 0; i < rows; ++i)
        for (int j = columns; j < columns + padding; ++j)
            EXPECT_EQ(C(i, j), value_t(123)) << "Output padding was overwritten";

#ifdef DEBUG_MATRIXMULTIPLICATION
    (void)max_error_ratio;
#else
    EXPECT_LE(max_error_ratio, 1.0L);
#endif
}
}  // namespace

TEST(Math, MatrixMultiplication)
{
    const int rows    = 64 + 32 + 8 + 4 + 3;
    const int lda     = 64 + 3;
    const int columns = 64 + 32 + 8 + 4 + 1 + 9;

    linalg_test::for_each_available_backend([&](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        matrix_vector_multiplication<double, Backend>(rows, columns);
        matrix_vector_multiplication<float, Backend>(rows, columns);

        vector_matrix_multiplication<float, Backend>(rows, columns);
        vector_matrix_multiplication<double, Backend>(rows, columns);

        matrix_multiplication_test<float, Backend>(rows, lda, columns, false, false);
        matrix_multiplication_test<float, Backend>(rows, lda, columns, true, false);
        matrix_multiplication_test<float, Backend>(rows, lda, columns, false, true);
        matrix_multiplication_test<float, Backend>(rows, lda, columns, true, true);

        matrix_multiplication_test<double, Backend>(rows, lda, columns, false, false);
        matrix_multiplication_test<double, Backend>(rows, lda, columns, true, false);
        matrix_multiplication_test<double, Backend>(rows, lda, columns, false, true);
        matrix_multiplication_test<double, Backend>(rows, lda, columns, true, true);
    });
}

// The scalar backend's LINALG_HAS_VECTORIZATION path (multiplication.cpp)
// packs operands into mr/nr-sized register blocks (mr = 3 * linalg::simd<
// value_t>::size, i.e. 12 for float, 6 for double; nr = 4) and only
// activates once rows + columns + depth > 20 — below that it falls through
// to the same plain scalar loop as the non-vectorized build. This sweeps
// depth/columns from 1 to 16 so every case lands on both sides of that
// threshold and at every remainder relative to the packet width (4 for
// float, 2 for double) and the mr/nr block sizes above, on every pack_lhs/
// pack_rhs/gemm peeling branch.
TEST(Math, MatrixMultiplicationVectorizedBoundary)
{
    constexpr int rows = 5;

    linalg_test::for_each_available_backend([&](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        for (int columns = 1; columns <= 16; ++columns)
        {
            for (int depth = 1; depth <= 16; ++depth)
            {
                matrix_multiplication_test<float, Backend>(rows, depth, columns, false, false);
                matrix_multiplication_test<float, Backend>(rows, depth, columns, true, false);
                matrix_multiplication_test<float, Backend>(rows, depth, columns, false, true);
                matrix_multiplication_test<float, Backend>(rows, depth, columns, true, true);

                matrix_multiplication_test<double, Backend>(rows, depth, columns, false, false);
                matrix_multiplication_test<double, Backend>(rows, depth, columns, true, false);
                matrix_multiplication_test<double, Backend>(rows, depth, columns, false, true);
                matrix_multiplication_test<double, Backend>(rows, depth, columns, true, true);
            }
        }
    });
}

// Row strides must be preserved by both the scalar and blocked GEMM paths.
TEST(Math, MatrixMultiplicationPaddedStrides)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            for (const int rows : {1, 5, 7, 33})
                for (const int depth : {3, 9})
                    for (const bool ta : {false, true})
                        for (const bool tb : {false, true})
                        {
                            matrix_multiplication_test<float, Backend>(rows, 7, depth, ta, tb, 2);
                            matrix_multiplication_test<double, Backend>(rows, 7, depth, ta, tb, 2);
                        }
        });
}

// alpha/beta must be honoured on both the scalar fallback (rows + columns +
// depth <= 20) and the blocked GEBP path, for every transpose combination,
// including beta == 1 (pure accumulate) and negative scales.
TEST(Math, MatrixMultiplicationAlphaBeta)
{
    struct scales
    {
        double alpha;
        double beta;
    };
    constexpr scales cases[] = {
        {2.5, 0.0}, {1.0, 1.0}, {-0.5, 1.5}, {1.25, -1.0}, {-1.0, 0.25}};

    linalg_test::for_each_available_backend(
        [&](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            for (const auto& s : cases)
                for (const bool ta : {false, true})
                    for (const bool tb : {false, true})
                    {
                        // Scalar fallback.
                        matrix_multiplication_test<float, Backend>(
                            5, 4, 3, ta, tb, 1, (float)s.alpha, (float)s.beta);
                        matrix_multiplication_test<double, Backend>(
                            5, 4, 3, ta, tb, 1, s.alpha, s.beta);
                        // Blocked path, with remainders on every peeling branch.
                        matrix_multiplication_test<float, Backend>(
                            111, 67, 118, ta, tb, 2, (float)s.alpha, (float)s.beta);
                        matrix_multiplication_test<double, Backend>(
                            111, 67, 118, ta, tb, 2, s.alpha, s.beta);
                    }
        });
}

// BLAS ?gemm contract: beta == 0 makes c write-only, so NaN already in c must
// not reach the result; alpha == 0 skips the product, so a and b are not read
// (NaN there must not reach the result) and c is only scaled by beta.
template <typename value_t, linalg::linalg_backend Backend>
void gemm_special_scales_test(int rows, int columns, int depth)
{
    SCOPED_TRACE(::testing::Message()
                 << "backend=" << static_cast<int>(Backend) << ", scalar bytes=" << sizeof(value_t)
                 << ", rows=" << rows << ", columns=" << columns << ", depth=" << depth);
    const value_t nan = std::numeric_limits<value_t>::quiet_NaN();
    auto          ws  = gemm_workspace<value_t, Backend>(rows, columns, depth);

    // beta == 0 with a NaN-filled output.
    {
        dense_matrix<value_t> A(rows, depth);
        dense_matrix<value_t> B(depth, columns);
        dense_matrix<value_t> C(rows, columns);
        A.fill(value_t(1));
        B.fill(value_t(2));
        C.fill(nan);
        linalg::multiplication<value_t, Backend>::apply(false,
            false,
            rows,
            columns,
            depth,
            value_t(3),
            A.data(),
            depth,
            B.data(),
            columns,
            value_t(0),
            C.data(),
            columns,
            ws.data(),
            ws.size());
        const value_t expected = value_t(3) * value_t(2) * static_cast<value_t>(depth);
        for (const value_t c : C)
        {
            ASSERT_EQ(c, expected);
        }
    }

    // alpha == 0 with NaN-filled operands.
    {
        dense_matrix<value_t> A(rows, depth);
        dense_matrix<value_t> B(depth, columns);
        dense_matrix<value_t> C(rows, columns);
        A.fill(nan);
        B.fill(nan);
        C.fill(value_t(4));
        linalg::multiplication<value_t, Backend>::apply(false,
            false,
            rows,
            columns,
            depth,
            value_t(0),
            A.data(),
            depth,
            B.data(),
            columns,
            value_t(-0.5),
            C.data(),
            columns,
            ws.data(),
            ws.size());
        for (const value_t c : C)
        {
            ASSERT_EQ(c, value_t(-2));
        }
    }
}

TEST(Math, MatrixMultiplicationSpecialScales)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            gemm_special_scales_test<float, Backend>(3, 4, 5);
            gemm_special_scales_test<double, Backend>(3, 4, 5);
            gemm_special_scales_test<float, Backend>(70, 50, 40);
            gemm_special_scales_test<double, Backend>(70, 50, 40);
        });
}

// Explicit validation of GEMM formula: C = alpha * op(A) * op(B) + beta * C
// with specific, non-trivial alpha/beta values and known input matrices.
template <typename value_t, linalg::linalg_backend Backend>
void gemm_formula_validation_test(value_t alpha, value_t beta)
{
    SCOPED_TRACE(::testing::Message()
                 << "backend=" << static_cast<int>(Backend) << ", scalar bytes=" << sizeof(value_t)
                 << ", alpha=" << alpha << ", beta=" << beta);

    constexpr int rows    = 3;
    constexpr int columns = 4;
    constexpr int depth   = 5;

    dense_matrix<value_t> A(rows, depth);
    dense_matrix<value_t> B(depth, columns);
    dense_matrix<value_t> C(rows, columns);
    dense_matrix<value_t> C_initial(rows, columns);

    auto ws = gemm_workspace<value_t, Backend>(rows, columns, depth);

    auto                                   generator = test_generator();
    std::uniform_real_distribution<double> distribution(-3., 3.);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < depth; ++j)
        {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                (value_t)distribution(generator);
        }
    }

    for (int i = 0; i < depth; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            B(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                (value_t)distribution(generator);
        }
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            C_initial(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                (value_t)distribution(generator);
            C(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                C_initial(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }

    linalg::multiplication<value_t, Backend>::apply(false,
        false,
        rows,
        columns,
        depth,
        alpha,
        A.data(),
        depth,
        B.data(),
        columns,
        beta,
        C.data(),
        columns,
        ws.data(),
        ws.size());

    constexpr auto tol       = matrixTolerance<value_t>::tolerance;
    const long double roundoff = std::numeric_limits<value_t>::epsilon() / 2.0L;
    const long double steps    = 2.0L * depth + 4.0L;
    const long double gamma    = (steps * roundoff) / (1.0L - steps * roundoff);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            long double product = 0;
            long double abs_product = 0;
            for (int k = 0; k < depth; ++k)
            {
                const long double ab = static_cast<long double>(A(static_cast<std::size_t>(i), static_cast<std::size_t>(k))) *
                                       static_cast<long double>(B(static_cast<std::size_t>(k), static_cast<std::size_t>(j)));
                product += ab;
                abs_product += std::fabs(ab);
            }

            const long double prior = static_cast<long double>(beta) *
                                      static_cast<long double>(C_initial(
                                          static_cast<std::size_t>(i), static_cast<std::size_t>(j)));
            const long double expected =
                static_cast<long double>(alpha) * product + prior;
            const long double magnitude =
                std::fabs(static_cast<long double>(alpha)) * abs_product + std::fabs(prior);

            const auto c_val = C(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            ASSERT_TRUE(std::isfinite(c_val))
                << "row=" << i << ", column=" << j << ", alpha=" << alpha << ", beta=" << beta;

            const long double allowed_error = std::max(static_cast<long double>(tol), gamma * magnitude);
            const long double error = std::fabs(expected - static_cast<long double>(c_val));
            EXPECT_LE(error, allowed_error)
                << "row=" << i << ", column=" << j << ", alpha=" << alpha << ", beta=" << beta
                << "\n  expected: " << expected << "\n  actual: " << c_val
                << "\n  allowed_error: " << allowed_error;
        }
    }
}

TEST(Math, MatrixMultiplicationGemmFormula)
{
    struct alpha_beta_pair
    {
        double alpha;
        double beta;
    };

    constexpr alpha_beta_pair pairs[] = {
        {2.0, 0.5},
        {0.5, 2.0},
        {-1.5, 1.5},
        {3.0, -2.0},
        {-0.25, 0.75},
    };

    linalg_test::for_each_available_backend(
        [&](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            for (const auto& pair : pairs)
            {
                gemm_formula_validation_test<float, Backend>((float)pair.alpha, (float)pair.beta);
                gemm_formula_validation_test<double, Backend>(pair.alpha, pair.beta);
            }
        });
}
