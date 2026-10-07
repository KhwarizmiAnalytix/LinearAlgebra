#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "include/matrix_operation/pentadiagonal_operations.h"
#include "include/matrix_operation/tridiagonal_operations.h"

namespace
{
using namespace linalg;

constexpr double EPSILON = 1e-10;

// Helper function to generate random-like test data
void fill_test_data(std::vector<double>& vec, size_t seed = 1)
{
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = 0.1 * (1.0 + ((i + seed) % 10));
    }
}

// Test tridiagonal decomposition and solve
TEST(MultiDiagonalOperationsTest, TridiagonalDecompositionAndSolve)
{
    const size_t outer_dim = 2;
    const size_t dim       = 4;
    const size_t inner_dim = 3;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    std::vector<double> decomposed(3 * stride);
    std::vector<double> x(stride);
    std::vector<double> result(stride);

    fill_test_data(decomposed, 1);
    fill_test_data(x, 2);
    result = x;

    EXPECT_NO_THROW(tridiagonal_operations::decomposition(
        decomposed, outer_dim, dim, inner_dim, false));
    EXPECT_NO_THROW(
        tridiagonal_operations::solve_decomposed(x, decomposed, outer_dim, dim, inner_dim, false));
}

// Test tridiagonal multiply
TEST(MultiDiagonalOperationsTest, TridiagonalMultiply)
{
    const size_t outer_dim = 2;
    const size_t dim       = 4;
    const size_t inner_dim = 3;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;
    double      time_multiplier = 1.0;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride);
    std::vector<double> mat(3 * stride);

    fill_test_data(in, 1);
    fill_test_data(mat, 2);

    EXPECT_NO_THROW(tridiagonal_operations::multiply(
        result, in, mat, outer_dim, dim, inner_dim, time_multiplier, false));

    // Verify that multiply produced non-zero results
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

// Test pentadiagonal decomposition and solve
TEST(MultiDiagonalOperationsTest, PentadiagonalDecompositionAndSolve)
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

    EXPECT_NO_THROW(pentadiagonal_operations::decomposition(
        decomposed, outer_dim, dim, inner_dim, false));
    EXPECT_NO_THROW(
        pentadiagonal_operations::solve_decomposed(x, decomposed, outer_dim, dim, inner_dim, false));
}

// Test pentadiagonal multiply
TEST(MultiDiagonalOperationsTest, PentadiagonalMultiply)
{
    const size_t outer_dim = 2;
    const size_t dim       = 4;
    const size_t inner_dim = 3;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;
    double      time_multiplier = 0.5;

    std::vector<double> result(stride, 0.0);
    std::vector<double> in(stride);
    std::vector<double> mat(5 * stride);

    fill_test_data(in, 1);
    fill_test_data(mat, 2);

    EXPECT_NO_THROW(pentadiagonal_operations::multiply(
        result, in, mat, outer_dim, dim, inner_dim, time_multiplier, false));

    // Verify that multiply produced non-zero results
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

// Test tridiagonal with various dimensions
TEST(MultiDiagonalOperationsTest, TridiagonalVaryingDimensions)
{
    std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
        {1, 8, 1},    // outer_dim=1, dim=8, inner_dim=1
        {4, 2, 5},    // outer_dim=4, dim=2, inner_dim=5
        {2, 3, 2},    // outer_dim=2, dim=3, inner_dim=2
    };

    for (const auto& [outer_dim, dim, inner_dim] : test_cases)
    {
        const size_t n      = dim * inner_dim;
        const size_t stride = outer_dim * n;

        std::vector<double> decomposed(3 * stride);
        fill_test_data(decomposed, 1);

        EXPECT_NO_THROW(
            tridiagonal_operations::decomposition(decomposed, outer_dim, dim, inner_dim, false))
            << "Failed for outer_dim=" << outer_dim << ", dim=" << dim << ", inner_dim=" << inner_dim;
    }
}

// Test pentadiagonal with various dimensions
TEST(MultiDiagonalOperationsTest, PentadiagonalVaryingDimensions)
{
    std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
        {1, 8, 1},    // outer_dim=1, dim=8, inner_dim=1
        {4, 2, 5},    // outer_dim=4, dim=2, inner_dim=5
        {2, 3, 2},    // outer_dim=2, dim=3, inner_dim=2
    };

    for (const auto& [outer_dim, dim, inner_dim] : test_cases)
    {
        const size_t n      = dim * inner_dim;
        const size_t stride = outer_dim * n;

        std::vector<double> decomposed(5 * stride);
        fill_test_data(decomposed, 1);

        EXPECT_NO_THROW(
            pentadiagonal_operations::decomposition(decomposed, outer_dim, dim, inner_dim, false))
            << "Failed for outer_dim=" << outer_dim << ", dim=" << dim << ", inner_dim=" << inner_dim;
    }
}

// Test error handling for wrong sizes
TEST(MultiDiagonalOperationsTest, TridiagonalWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;
    const size_t n         = dim * inner_dim;
    const size_t stride    = outer_dim * n;

    // Wrong buffer size (should be 3*stride for tridiagonal)
    std::vector<double> decomposed(2 * stride);

    EXPECT_THROW(
        tridiagonal_operations::decomposition(decomposed, outer_dim, dim, inner_dim, false),
        std::invalid_argument);
}

// Test error handling for wrong sizes in pentadiagonal
TEST(MultiDiagonalOperationsTest, PentadiagonalWrongSize)
{
    const size_t outer_dim = 1;
    const size_t dim       = 2;
    const size_t inner_dim = 1;

    // Wrong buffer size (should be 5*stride for pentadiagonal)
    std::vector<double> decomposed(3);

    EXPECT_THROW(
        pentadiagonal_operations::decomposition(decomposed, outer_dim, dim, inner_dim, false),
        std::invalid_argument);
}

}  // namespace
