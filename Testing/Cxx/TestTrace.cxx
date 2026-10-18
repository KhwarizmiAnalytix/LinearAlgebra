#include <cmath>
#include <cstdlib>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/trace.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct TraceTolerance
{
};

template <> struct TraceTolerance<float>
{
    static constexpr double tolerance = 1e-5;
};

template <> struct TraceTolerance<double>
{
    static constexpr double tolerance = 1e-12;
};

template <typename value_t, linalg::linalg_backend Backend>
void test_trace(std::size_t n)
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
        linalg::trace<value_t, Backend>::apply(
            M.begin(), static_cast<linalg_int>(n), static_cast<linalg_int>(n));

    EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected),
        TraceTolerance<value_t>::tolerance);
}
}  // namespace

TEST(Math, MatrixTrace)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_trace<float, Backend>(1);
        test_trace<float, Backend>(5);
        test_trace<double, Backend>(1);
        test_trace<double, Backend>(7);
    });
}
