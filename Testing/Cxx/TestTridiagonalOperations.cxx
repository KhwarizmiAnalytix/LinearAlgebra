#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "include/matrix_operation/tridiagonal_operations.h"

namespace
{
using linalg::tridiagonal_operations::decomposition;
using linalg::tridiagonal_operations::solve_decomposed;
using linalg::tridiagonal_operations::solve_tridiagonal;
using linalg::tridiagonal_operations::multiply;

constexpr double EPSILON = 1e-10;

TEST(TridiagonalOperations, DecompositionBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> output(3 * stride, 1.0);

    EXPECT_NO_THROW(decomposition(output, outer_dim, dim, inner_dim, false));
    EXPECT_EQ(output.size(), 3 * stride);
}

TEST(TridiagonalOperations, DecompositionWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;

    std::vector<double> output(2, 1.0);

    EXPECT_THROW(decomposition(output, outer_dim, dim, inner_dim, false), std::invalid_argument);
}

TEST(TridiagonalOperations, SolveTridiagonal)
{
    const size_t n = 3;
    std::vector<double> L(n, 0.5);
    std::vector<double> D(n, 2.0);
    std::vector<double> U(n, 0.5);
    std::vector<double> output(n, 1.0);

    EXPECT_NO_THROW(solve_tridiagonal(output.data(), n, L.data(), D.data(), U.data()));
}

TEST(TridiagonalOperations, SolveTridiagonalSmallN)
{
    const size_t n = 2;
    std::vector<double> L(n, 0.5);
    std::vector<double> D(n, 2.0);
    std::vector<double> U(n, 0.5);
    std::vector<double> output(n, 1.0);

    EXPECT_THROW(solve_tridiagonal(output.data(), n, L.data(), D.data(), U.data()),
        std::invalid_argument);
}

TEST(TridiagonalOperations, MultiplyBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride, 1.0);
    std::vector<double> mat(3 * stride, 0.1);

    EXPECT_NO_THROW(multiply(result, in, mat, outer_dim, dim, inner_dim, 0.5, false));

    for (size_t i = 0; i < stride; ++i)
    {
        EXPECT_GT(result[i], 0.0);
    }
}

TEST(TridiagonalOperations, SolveDecomposedBasic)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(3 * stride, 0.1);
    std::vector<double> x(stride, 1.0);

    EXPECT_NO_THROW(solve_decomposed(x, decomposed, outer_dim, dim, inner_dim, false));
}

}  // namespace
