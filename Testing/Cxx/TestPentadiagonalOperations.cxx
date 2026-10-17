#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "test_reporting.h"
#include "include/matrix_operation/banded_operations.h"

namespace
{
using namespace linalg::banded_operations;

constexpr double EPSILON = 1e-10;

TEST(BandedOperations5, DecompositionBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> output(5 * stride, 1.0);

    EXPECT_NO_THROW((decomposition<double, 5>(output, outer_dim, dim, inner_dim)));
    EXPECT_EQ(output.size(), 5 * stride);
}

TEST(BandedOperations5, DecompositionBasicFloat)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<float> output(5 * stride, 1.0f);

    EXPECT_NO_THROW((decomposition<float, 5>(output, outer_dim, dim, inner_dim)));
    EXPECT_EQ(output.size(), 5 * stride);
}

TEST(BandedOperations5, DecompositionWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;

    std::vector<double> output(10, 1.0);

    EXPECT_THROW((decomposition<double, 5>(output, outer_dim, dim, inner_dim)),
        std::invalid_argument);
}

TEST(BandedOperations5, SolveDecomposedBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(5 * stride, 0.1);
    std::vector<double> x(stride, 1.0);

    EXPECT_NO_THROW((solve_decomposed<double, 5>(x, decomposed, outer_dim, dim, inner_dim)));
}

TEST(BandedOperations5, SolveDecomposedBasicFloat)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<float> decomposed(5 * stride, 0.1f);
    std::vector<float> x(stride, 1.0f);

    EXPECT_NO_THROW((solve_decomposed<float, 5>(x, decomposed, outer_dim, dim, inner_dim)));
}

TEST(BandedOperations5, SolveDecomposedWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(5 * stride, 0.1);
    std::vector<double> x(stride - 1, 1.0);

    EXPECT_THROW((solve_decomposed<double, 5>(x, decomposed, outer_dim, dim, inner_dim)),
        std::invalid_argument);
}

TEST(BandedOperations5, MultiplyBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride, 1.0);
    std::vector<double> mat(5 * stride, 0.1);

    EXPECT_NO_THROW((multiply<double, 5>(result, in, mat, outer_dim, dim, inner_dim, 0.5)));

    for (size_t i = 0; i < stride; ++i)
    {
        EXPECT_GT(result[i], 0.0);
    }
}

TEST(BandedOperations5, MultiplyBasicFloat)
{
    const size_t outer_dim = 2;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<float> result(stride, 0.0f);
    std::vector<float> in(stride, 1.0f);
    std::vector<float> mat(5 * stride, 0.1f);

    EXPECT_NO_THROW((multiply<float, 5>(result, in, mat, outer_dim, dim, inner_dim, 0.5f)));

    for (size_t i = 0; i < stride; ++i)
    {
        EXPECT_GT(result[i], 0.0f);
    }
}

}  // namespace
