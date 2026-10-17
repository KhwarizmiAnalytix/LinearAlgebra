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

void fill_test_data(std::vector<double>& vec, size_t seed = 1)
{
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = 0.1 * (1.0 + ((i + seed) % 10));
    }
}

TEST(BandedOperations, TridiagonalDecompositionAndSolve)
{
    const size_t outer_dim = 2;
    const size_t dim       = 4;
    const size_t inner_dim = 3;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(3 * stride);
    std::vector<double> x(stride);

    fill_test_data(decomposed, 1);
    fill_test_data(x, 2);

    EXPECT_NO_THROW((decomposition<double, 3>(decomposed, outer_dim, dim, inner_dim)));
    EXPECT_NO_THROW((solve_decomposed<double, 3>(x, decomposed, outer_dim, dim, inner_dim)));
}

TEST(BandedOperations, TridiagonalMultiply)
{
    const size_t outer_dim      = 2;
    const size_t dim            = 4;
    const size_t inner_dim      = 3;
    const size_t n              = dim * inner_dim;
    const size_t stride         = outer_dim * n;
    double       time_multiplier = 1.0;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride);
    std::vector<double> mat(3 * stride);

    fill_test_data(in, 1);
    fill_test_data(mat, 2);

    EXPECT_NO_THROW(
        (multiply<double, 3>(result, in, mat, outer_dim, dim, inner_dim, time_multiplier)));

    bool has_nonzero = false;
    for (size_t i = 0; i < stride; ++i)
    {
        if (std::abs(result[i]) > EPSILON)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(BandedOperations, PentadiagonalDecompositionAndSolve)
{
    const size_t outer_dim = 2;
    const size_t dim       = 4;
    const size_t inner_dim = 3;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(5 * stride);
    std::vector<double> x(stride);

    fill_test_data(decomposed, 1);
    fill_test_data(x, 2);

    EXPECT_NO_THROW((decomposition<double, 5>(decomposed, outer_dim, dim, inner_dim)));
    EXPECT_NO_THROW((solve_decomposed<double, 5>(x, decomposed, outer_dim, dim, inner_dim)));
}

TEST(BandedOperations, PentadiagonalMultiply)
{
    const size_t outer_dim      = 2;
    const size_t dim            = 4;
    const size_t inner_dim      = 3;
    const size_t n              = dim * inner_dim;
    const size_t stride         = outer_dim * n;
    double       time_multiplier = 0.5;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride);
    std::vector<double> mat(5 * stride);

    fill_test_data(in, 1);
    fill_test_data(mat, 2);

    EXPECT_NO_THROW(
        (multiply<double, 5>(result, in, mat, outer_dim, dim, inner_dim, time_multiplier)));

    bool has_nonzero = false;
    for (size_t i = 0; i < stride; ++i)
    {
        if (std::abs(result[i]) > EPSILON)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(BandedOperations, TridiagonalVaryingDimensions)
{
    std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
        {1, 8, 1},
        {4, 2, 5},
        {2, 3, 2},
    };

    for (const auto& tc : test_cases)
    {
        const size_t outer_dim = std::get<0>(tc);
        const size_t dim       = std::get<1>(tc);
        const size_t inner_dim = std::get<2>(tc);
        const size_t n         = dim * inner_dim;
        const size_t stride    = outer_dim * n;

        std::vector<double> decomposed(3 * stride);
        fill_test_data(decomposed, 1);

        EXPECT_NO_THROW((decomposition<double, 3>(decomposed, outer_dim, dim, inner_dim)))
            << "Failed for outer_dim=" << outer_dim << ", dim=" << dim
            << ", inner_dim=" << inner_dim;
    }
}

TEST(BandedOperations, PentadiagonalVaryingDimensions)
{
    std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
        {1, 8, 1},
        {4, 2, 5},
        {2, 3, 2},
    };

    for (const auto& tc : test_cases)
    {
        const size_t outer_dim = std::get<0>(tc);
        const size_t dim       = std::get<1>(tc);
        const size_t inner_dim = std::get<2>(tc);
        const size_t n         = dim * inner_dim;
        const size_t stride    = outer_dim * n;

        std::vector<double> decomposed(5 * stride);
        fill_test_data(decomposed, 1);

        EXPECT_NO_THROW((decomposition<double, 5>(decomposed, outer_dim, dim, inner_dim)))
            << "Failed for outer_dim=" << outer_dim << ", dim=" << dim
            << ", inner_dim=" << inner_dim;
    }
}

TEST(BandedOperations, TridiagonalWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(2 * stride);

    EXPECT_THROW((decomposition<double, 3>(decomposed, outer_dim, dim, inner_dim)),
        std::invalid_argument);
}

TEST(BandedOperations, PentadiagonalWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;

    std::vector<double> decomposed(3);

    EXPECT_THROW((decomposition<double, 5>(decomposed, outer_dim, dim, inner_dim)),
        std::invalid_argument);
}

}  // namespace
