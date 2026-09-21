#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/qr_decomposition.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t> void test_qr(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            A(i, j) = static_cast<value_t>(20. * rand() / RAND_MAX - 10.);
        }
    }

    const std::size_t k = std::min(rows, columns);

    dense_matrix<value_t> Q(rows, k);
    dense_matrix<value_t> R(k, columns);

    linalg::qr_decomposition(
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        A.begin(),
        static_cast<linalg_long>(columns),
        Q.begin(),
        static_cast<linalg_long>(k),
        R.begin(),
        static_cast<linalg_long>(columns));

    // Q*R should reconstruct A.
    value_t max_error = 0;
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            value_t sum = 0;
            for (std::size_t c = 0; c < k; c++)
            {
                sum += Q(i, c) * R(c, j);
            }
            max_error = std::max(max_error, static_cast<value_t>(std::fabs(A(i, j) - sum)));
        }
    }
    EXPECT_LT(max_error, static_cast<value_t>(1e-3));

    // Q should have orthonormal columns: Q^T * Q == I_k.
    value_t max_orth_error = 0;
    for (std::size_t i = 0; i < k; i++)
    {
        for (std::size_t j = 0; j < k; j++)
        {
            value_t sum = 0;
            for (std::size_t r = 0; r < rows; r++)
            {
                sum += Q(r, i) * Q(r, j);
            }
            const value_t expected      = (i == j) ? value_t(1) : value_t(0);
            max_orth_error = std::max(max_orth_error, static_cast<value_t>(std::fabs(sum - expected)));
        }
    }
    EXPECT_LT(max_orth_error, static_cast<value_t>(1e-3));

    // R should be upper-triangular (zero strictly below the diagonal).
    for (std::size_t i = 0; i < k; i++)
    {
        for (std::size_t j = 0; j < i; j++)
        {
            EXPECT_EQ(R(i, j), value_t(0));
        }
    }
}
}  // namespace

TEST(Math, QRDecomposition)
{
    test_qr<float>(6, 5);
    test_qr<float>(5, 6);
    test_qr<float>(4, 4);
    test_qr<double>(6, 5);
    test_qr<double>(5, 6);
    test_qr<double>(4, 4);
}
