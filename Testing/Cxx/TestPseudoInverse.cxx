#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/pseudo_inverse.h"

namespace
{
using linalg_test::dense_matrix;

// Full column-rank tall matrix: pinv(A) is a left inverse, pinv(A)*A == I.
template <typename value_t> void test_pseudo_inverse_tall(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            A(i, j) = static_cast<value_t>(20. * rand() / RAND_MAX - 10.);
        }
    }

    dense_matrix<value_t> Ainv(columns, rows);
    linalg::pseudo_inverse(
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        A.begin(),
        static_cast<linalg_long>(columns),
        Ainv.begin(),
        static_cast<linalg_long>(rows));

    value_t max_error = 0;
    for (std::size_t i = 0; i < columns; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < rows; k++)
            {
                sum += Ainv(i, k) * A(k, j);
            }
            const value_t expected = (i == j) ? value_t(1) : value_t(0);
            max_error               = std::max(max_error, static_cast<value_t>(std::fabs(sum - expected)));
        }
    }
    EXPECT_LT(max_error, static_cast<value_t>(1e-2));
}
}  // namespace

TEST(Math, PseudoInverse)
{
    test_pseudo_inverse_tall<float>(6, 4);
    test_pseudo_inverse_tall<double>(6, 4);
    test_pseudo_inverse_tall<float>(9, 3);
    test_pseudo_inverse_tall<double>(9, 3);
}
