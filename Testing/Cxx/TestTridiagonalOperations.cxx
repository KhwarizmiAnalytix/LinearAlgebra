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

TEST(BandedOperations3, DecompositionBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> output(3 * stride, 1.0);

    EXPECT_NO_THROW((decomposition<double, 3>(output, outer_dim, dim, inner_dim)));
    EXPECT_EQ(output.size(), 3 * stride);
}

TEST(BandedOperations3, DecompositionBasicFloat)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<float> output(3 * stride, 1.0f);

    EXPECT_NO_THROW((decomposition<float, 3>(output, outer_dim, dim, inner_dim)));
    EXPECT_EQ(output.size(), 3 * stride);
}

TEST(BandedOperations3, DecompositionWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;

    std::vector<double> output(2, 1.0);

    EXPECT_THROW((decomposition<double, 3>(output, outer_dim, dim, inner_dim)),
        std::invalid_argument);
}

TEST(BandedOperations3, SolveBanded)
{
    const size_t n = 3;
    std::vector<double> L(n, 0.5);
    std::vector<double> D(n, 2.0);
    std::vector<double> U(n, 0.5);
    std::vector<double> output(n, 1.0);

    EXPECT_NO_THROW(solve_banded(output.data(), n, L.data(), D.data(), U.data()));
}

TEST(BandedOperations3, SolveBandedSmallN)
{
    const size_t n = 2;
    std::vector<double> L(n, 0.5);
    std::vector<double> D(n, 2.0);
    std::vector<double> U(n, 0.5);
    std::vector<double> output(n, 1.0);

    EXPECT_THROW(solve_banded(output.data(), n, L.data(), D.data(), U.data()),
        std::invalid_argument);
}

TEST(BandedOperations3, SolveBandedFloat)
{
    const size_t n = 3;
    std::vector<float> L(n, 0.5f);
    std::vector<float> D(n, 2.0f);
    std::vector<float> U(n, 0.5f);
    std::vector<float> output(n, 1.0f);

    EXPECT_NO_THROW(solve_banded(output.data(), n, L.data(), D.data(), U.data()));
}

TEST(BandedOperations3, MultiplyBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride, 1.0);
    std::vector<double> mat(3 * stride, 0.1);

    EXPECT_NO_THROW((multiply<double, 3>(result, in, mat, outer_dim, dim, inner_dim, 0.5)));

    for (size_t i = 0; i < stride; ++i)
    {
        EXPECT_GT(result[i], 0.0);
    }
}

TEST(BandedOperations3, MultiplyBasicFloat)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<float> result(stride, 0.0f);
    std::vector<float> in(stride, 1.0f);
    std::vector<float> mat(3 * stride, 0.1f);

    EXPECT_NO_THROW((multiply<float, 3>(result, in, mat, outer_dim, dim, inner_dim, 0.5f)));

    for (size_t i = 0; i < stride; ++i)
    {
        EXPECT_GT(result[i], 0.0f);
    }
}

TEST(BandedOperations3, SolveDecomposedBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(3 * stride, 0.1);
    std::vector<double> x(stride, 1.0);

    EXPECT_NO_THROW((solve_decomposed<double, 3>(x, decomposed, outer_dim, dim, inner_dim)));
}

}  // namespace
