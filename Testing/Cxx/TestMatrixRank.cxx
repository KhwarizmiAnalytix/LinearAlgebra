#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/matrix_rank.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t> void test_rank_full(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            A(i, j) = static_cast<value_t>(20. * rand() / RAND_MAX - 10.);
        }
    }

    const auto rank = linalg::matrix_rank(
        A.begin(),
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(columns));
    EXPECT_EQ(rank, static_cast<linalg_long>(std::min(rows, columns)));
}

// Rank-1 matrix built as an outer product: u * v^T always has rank <= 1.
template <typename value_t> void test_rank_deficient(std::size_t n)
{
    dense_matrix<value_t> u(n, 1);
    dense_matrix<value_t> v(n, 1);
    for (std::size_t i = 0; i < n; i++)
    {
        u(i, 0) = static_cast<value_t>(1. + rand() / (double)RAND_MAX);
        v(i, 0) = static_cast<value_t>(1. + rand() / (double)RAND_MAX);
    }

    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            A(i, j) = u(i, 0) * v(j, 0);
        }
    }

    const auto rank = linalg::matrix_rank(
        A.begin(), static_cast<linalg_long>(n), static_cast<linalg_long>(n), static_cast<linalg_long>(n));
    EXPECT_EQ(rank, static_cast<linalg_long>(1));
}

template <typename value_t> void test_condition_number_identity(std::size_t n)
{
    dense_matrix<value_t> I(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        I(i, i) = value_t(1);
    }
    const auto cond = linalg::matrix_condition_number(
        I.begin(), static_cast<linalg_long>(n), static_cast<linalg_long>(n), static_cast<linalg_long>(n));
    EXPECT_NEAR(static_cast<double>(cond), 1.0, 1e-3);
}
}  // namespace

TEST(Math, MatrixRank)
{
    test_rank_full<float>(6, 4);
    test_rank_full<double>(6, 4);
    test_rank_deficient<float>(5);
    test_rank_deficient<double>(5);
}

TEST(Math, MatrixConditionNumber)
{
    test_condition_number_identity<float>(5);
    test_condition_number_identity<double>(5);
}
