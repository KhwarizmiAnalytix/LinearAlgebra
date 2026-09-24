#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "include/matrix_operation/pentadiagonal_operations.h"

namespace
{
using linalg::pentadiagonal_operations::decomposition;
using linalg::pentadiagonal_operations::solve_decomposed;
using linalg::pentadiagonal_operations::multiply;

constexpr double EPSILON = 1e-10;

TEST(PentadiagonalOperations, DecompositionBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> output(5 * stride, 1.0);

    EXPECT_NO_THROW(decomposition(output, outer_dim, dim, inner_dim, false));
    EXPECT_EQ(output.size(), 5 * stride);
}

TEST(PentadiagonalOperations, DecompositionWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;

    std::vector<double> output(10, 1.0);

    EXPECT_THROW(decomposition(output, outer_dim, dim, inner_dim, false), std::invalid_argument);
}

TEST(PentadiagonalOperations, SolveDecomposedBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(5 * stride, 0.1);
    std::vector<double> x(stride, 1.0);

    EXPECT_NO_THROW(solve_decomposed(x, decomposed, outer_dim, dim, inner_dim, false));
}

TEST(PentadiagonalOperations, SolveDecomposedWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(5 * stride, 0.1);
    std::vector<double> x(stride - 1, 1.0);

    EXPECT_THROW(solve_decomposed(x, decomposed, outer_dim, dim, inner_dim, false),
        std::invalid_argument);
}

TEST(PentadiagonalOperations, MultiplyBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride, 1.0);
    std::vector<double> mat(5 * stride, 0.1);

    EXPECT_NO_THROW(multiply(result, in, mat, outer_dim, dim, inner_dim, 0.5, false));

    for (size_t i = 0; i < stride; ++i)
    {
        EXPECT_GT(result[i], 0.0);
    }
}

TEST(PentadiagonalOperations, MultiplyWithParallelization)
{
    const size_t outer_dim = 2;
    const size_t dim       = 3;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride, 1.0);
    std::vector<double> mat(5 * stride, 0.1);

    EXPECT_NO_THROW(multiply(result, in, mat, outer_dim, dim, inner_dim, 0.5, true));

    for (size_t i = 0; i < stride; ++i)
    {
        EXPECT_GT(result[i], 0.0);
    }
}

}  // namespace
