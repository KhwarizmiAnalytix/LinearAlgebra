#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/least_squares.h"

namespace
{
using linalg_test::dense_matrix;

// Overdetermined, consistent system (B = A * X_true exactly): the
// least-squares solution should recover X_true (residual is zero).
template <typename value_t> void test_least_squares_overdetermined()
{
    const std::size_t rows = 6, columns = 3, nrhs = 2;

    dense_matrix<value_t> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            A(i, j) = static_cast<value_t>(1. + (i * 3 + j * 7) % 11);
        }
    }

    dense_matrix<value_t> X_true(columns, nrhs);
    for (std::size_t i = 0; i < columns; i++)
    {
        for (std::size_t j = 0; j < nrhs; j++)
        {
            X_true(i, j) = static_cast<value_t>(1 + i + 2 * j);
        }
    }

    dense_matrix<value_t> B(rows, nrhs);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < nrhs; j++)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < columns; k++)
            {
                sum += A(i, k) * X_true(k, j);
            }
            B(i, j) = sum;
        }
    }

    dense_matrix<value_t> X(columns, nrhs);
    linalg::least_squares_solve(
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(nrhs),
        A.begin(),
        static_cast<linalg_long>(columns),
        B.begin(),
        static_cast<linalg_long>(nrhs),
        X.begin(),
        static_cast<linalg_long>(nrhs));

    value_t max_error = 0;
    for (std::size_t i = 0; i < columns; i++)
    {
        for (std::size_t j = 0; j < nrhs; j++)
        {
            max_error = std::max(max_error, static_cast<value_t>(std::fabs(X(i, j) - X_true(i, j))));
        }
    }
    EXPECT_LT(max_error, static_cast<value_t>(1e-2));
}

// Underdetermined system: any solution satisfying A*X = B is a valid
// least-squares (zero-residual) answer; least_squares_solve should at
// least satisfy the system, since it is exactly solvable when rows < columns
// and A has full row rank.
template <typename value_t> void test_least_squares_underdetermined()
{
    const std::size_t rows = 3, columns = 6, nrhs = 1;

    dense_matrix<value_t> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            A(i, j) = static_cast<value_t>(1. + (i * 5 + j * 2) % 7);
        }
    }

    dense_matrix<value_t> B(rows, nrhs);
    for (std::size_t i = 0; i < rows; i++)
    {
        B(i, 0) = static_cast<value_t>(3 + i);
    }

    dense_matrix<value_t> X(columns, nrhs);
    linalg::least_squares_solve(
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(nrhs),
        A.begin(),
        static_cast<linalg_long>(columns),
        B.begin(),
        static_cast<linalg_long>(nrhs),
        X.begin(),
        static_cast<linalg_long>(nrhs));

    value_t max_error = 0;
    for (std::size_t i = 0; i < rows; i++)
    {
        value_t sum = 0;
        for (std::size_t k = 0; k < columns; k++)
        {
            sum += A(i, k) * X(k, 0);
        }
        max_error = std::max(max_error, static_cast<value_t>(std::fabs(sum - B(i, 0))));
    }
    EXPECT_LT(max_error, static_cast<value_t>(1e-2));
}
}  // namespace

TEST(Math, LeastSquares)
{
    test_least_squares_overdetermined<float>();
    test_least_squares_overdetermined<double>();
    test_least_squares_underdetermined<float>();
    test_least_squares_underdetermined<double>();
}
