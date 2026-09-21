#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/matrix_multiplication.h"

// #define DEBUG_MATRIXMULTIPLICATION

namespace
{
using linalg_test::dense_matrix;

template <typename>
struct matrixTolerance
{
};

template <>
struct matrixTolerance<float>
{
    static constexpr float tolerance = (float)1.5e-4;
};

template <>
struct matrixTolerance<double>
{
    static constexpr double tolerance = 5.e-13;
};

// v (1 x rows) * A (rows x columns) -> r (1 x columns), via the raw
// matrix_multiplication(transpose_a=false, transpose_b=false, ...) entry
// point, treating v as a 1-row matrix.
template <typename value_t>
void vector_matrix_multiplication(int rows, int columns)
{
    constexpr auto tol = matrixTolerance<value_t>::tolerance;

    dense_matrix<value_t> A(rows, columns);
    std::vector<value_t>  v(rows);
    std::vector<value_t>  r(columns);

    std::default_random_engine             generator;
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

    linalg::matrix_multiplication(
        false,
        false,
        1,
        columns,
        rows,
        v.data(),
        rows,
        A.data(),
        columns,
        r.data(),
        columns);

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
        max_error = std::fmax(
            max_error,
            std::fabs(static_cast<double>(r[static_cast<std::size_t>(j)] - r_seq[static_cast<std::size_t>(j)])));
    }

#ifdef DEBUG_MATRIXMULTIPLICATION
    (void)max_error;
#else
    EXPECT_LT(max_error, tol);
#endif
}

// A (rows x columns) * v (columns x 1) -> r (rows x 1), plus a transpose
// sanity check (B = transpose(A) computed by hand, since matrix_transpose
// is not exercised by this test file — matches the original test scope).
template <typename value_t>
void matrix_vector_multiplication(int rows, int columns)
{
    constexpr auto tol = matrixTolerance<value_t>::tolerance;

    dense_matrix<value_t> A(rows, columns);
    std::vector<value_t>  v(columns);
    std::vector<value_t>  r(rows);

    std::default_random_engine             generator;
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

    linalg::matrix_multiplication(
        false,
        false,
        rows,
        1,
        columns,
        A.data(),
        columns,
        v.data(),
        1,
        r.data(),
        1);

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
            max_error = std::fmax(
                max_error,
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

// General C(rows x columns) = op(A) * op(B), depth is the shared dimension.
// A's storage shape is (rows x depth) when !transpose_a, else (depth x rows);
// B's storage shape is (depth x columns) when !transpose_b, else
// (columns x depth) — matching linalg::matrix_multiplication's row-major
// convention (see include/matrix_operation/matrix_multiplication.cxx).
template <typename value_t>
void matrix_multiplication_test(int rows, int columns, int depth, bool transpose_a, bool transpose_b)
{
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

    dense_matrix<value_t> A(a_rows, a_columns);
    dense_matrix<value_t> B(b_rows, b_columns);
    dense_matrix<value_t> C(rows, columns);

    std::default_random_engine             generator;
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

    linalg::matrix_multiplication(
        transpose_a,
        transpose_b,
        rows,
        columns,
        depth,
        A.data(),
        static_cast<linalg_int>(a_columns),
        B.data(),
        static_cast<linalg_int>(b_columns),
        C.data(),
        static_cast<linalg_int>(columns));

    value_t max_error = 0;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            value_t sum = 0;
            for (int k = 0; k < depth; ++k)
            {
                const value_t a_val = transpose_a
                                           ? A(static_cast<std::size_t>(k), static_cast<std::size_t>(i))
                                           : A(static_cast<std::size_t>(i), static_cast<std::size_t>(k));
                const value_t b_val = transpose_b
                                           ? B(static_cast<std::size_t>(j), static_cast<std::size_t>(k))
                                           : B(static_cast<std::size_t>(k), static_cast<std::size_t>(j));
                sum += a_val * b_val;
            }
            auto c = C(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(sum - c)));
        }
    }

#ifdef DEBUG_MATRIXMULTIPLICATION
    (void)max_error;
#else
    EXPECT_LE(max_error, tol);
#endif
}
}  // namespace

TEST(Math, MatrixMultiplication)
{
    const int rows    = 64 + 32 + 8 + 4 + 3;
    const int lda     = 64 + 3;
    const int columns = 64 + 32 + 8 + 4 + 1 + 9;

    matrix_vector_multiplication<double>(rows, columns);
    matrix_vector_multiplication<float>(rows, columns);

    vector_matrix_multiplication<float>(rows, columns);
    vector_matrix_multiplication<double>(rows, columns);

    matrix_multiplication_test<float>(rows, lda, columns, false, false);
    matrix_multiplication_test<float>(rows, lda, columns, true, false);
    matrix_multiplication_test<float>(rows, lda, columns, false, true);
    matrix_multiplication_test<float>(rows, lda, columns, true, true);

    matrix_multiplication_test<double>(rows, lda, columns, false, false);
    matrix_multiplication_test<double>(rows, lda, columns, true, false);
    matrix_multiplication_test<double>(rows, lda, columns, false, true);
    matrix_multiplication_test<double>(rows, lda, columns, true, true);
}
