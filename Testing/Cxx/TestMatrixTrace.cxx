#include <cmath>
#include <cstdlib>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/matrix_trace.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t> void test_trace(std::size_t n)
{
    dense_matrix<value_t> M(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            M(i, j) = static_cast<value_t>(10. * rand() / RAND_MAX);
        }
    }

    value_t expected = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        expected += M(i, i);
    }

    const value_t actual =
        linalg::matrix_trace(M.begin(), static_cast<linalg_int>(n), static_cast<linalg_int>(n));

    EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected), 1e-9);
}
}  // namespace

TEST(Math, MatrixTrace)
{
    test_trace<float>(1);
    test_trace<float>(5);
    test_trace<double>(1);
    test_trace<double>(7);
}
