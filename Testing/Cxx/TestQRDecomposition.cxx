#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/qr_decomposition.h"
#include "gtest/gtest.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t, linalg::linalg_backend Backend>
void test_qr(std::size_t rows, std::size_t columns)
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

    const auto ws_size = linalg::qr_decomposition<value_t, Backend>::workspace_size(
        static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> ws(ws_size);

    linalg::qr_decomposition<value_t, Backend>::apply(static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        A.begin(),
        static_cast<linalg_long>(columns),
        Q.begin(),
        static_cast<linalg_long>(k),
        R.begin(),
        static_cast<linalg_long>(columns),
        ws.data(),
        ws_size);

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
            const value_t expected = (i == j) ? value_t(1) : value_t(0);
            max_orth_error =
                std::max(max_orth_error, static_cast<value_t>(std::fabs(sum - expected)));
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
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr auto Backend = decltype(backend)::value;
        test_qr<float, Backend>(6, 5);
        test_qr<float, Backend>(5, 6);
        test_qr<float, Backend>(4, 4);
        test_qr<double, Backend>(6, 5);
        test_qr<double, Backend>(5, 6);
        test_qr<double, Backend>(4, 4);
    });
}
