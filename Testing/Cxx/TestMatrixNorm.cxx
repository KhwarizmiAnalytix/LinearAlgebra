#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/matrix_norm.h"
#include "include/matrix_operation/svd_decomposition.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t> void test_norms(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> M(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            M(i, j) = static_cast<value_t>(20. * rand() / RAND_MAX - 10.);
        }
    }

    // Frobenius: direct definition.
    double frob_expected = 0;
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            frob_expected += static_cast<double>(M(i, j)) * static_cast<double>(M(i, j));
        }
    }
    frob_expected = std::sqrt(frob_expected);

    const auto frob = linalg::matrix_norm(
        M.begin(),
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(columns),
        linalg::matrix_norm_type::FROBENIUS);
    EXPECT_NEAR(static_cast<double>(frob), frob_expected, 1e-3);

    // One-norm: max absolute column sum.
    double one_expected = 0;
    for (std::size_t j = 0; j < columns; j++)
    {
        double s = 0;
        for (std::size_t i = 0; i < rows; i++)
        {
            s += std::fabs(static_cast<double>(M(i, j)));
        }
        one_expected = std::max(one_expected, s);
    }
    const auto one = linalg::matrix_norm(
        M.begin(),
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(columns),
        linalg::matrix_norm_type::ONE);
    EXPECT_NEAR(static_cast<double>(one), one_expected, 1e-3);

    // Infinity-norm: max absolute row sum.
    double inf_expected = 0;
    for (std::size_t i = 0; i < rows; i++)
    {
        double s = 0;
        for (std::size_t j = 0; j < columns; j++)
        {
            s += std::fabs(static_cast<double>(M(i, j)));
        }
        inf_expected = std::max(inf_expected, s);
    }
    const auto inf = linalg::matrix_norm(
        M.begin(),
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(columns),
        linalg::matrix_norm_type::INFINITY_NORM);
    EXPECT_NEAR(static_cast<double>(inf), inf_expected, 1e-3);

    // Two-norm: cross-check against the largest singular value from
    // svd_decomposition directly.
    dense_matrix<value_t> A_copy(rows, columns);
    for (std::size_t i = 0; i < rows * columns; i++)
    {
        A_copy.begin()[i] = M.begin()[i];
    }
    std::size_t            k = std::min(rows, columns);
    std::vector<value_t>   S(k);
    dense_matrix<value_t>  U(rows, k);
    dense_matrix<value_t>  VT(k, columns);
    linalg::svd_decomposition(
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        A_copy.begin(),
        static_cast<linalg_long>(columns),
        S.data(),
        U.begin(),
        static_cast<linalg_long>(k),
        VT.begin(),
        static_cast<linalg_long>(columns));
    value_t smax = 0;
    for (auto s : S)
    {
        smax = std::max(smax, static_cast<value_t>(std::fabs(s)));
    }

    const auto two = linalg::matrix_norm(
        M.begin(),
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(columns),
        linalg::matrix_norm_type::TWO);
    EXPECT_NEAR(static_cast<double>(two), static_cast<double>(smax), 1e-3);
}
}  // namespace

TEST(Math, MatrixNorm)
{
    test_norms<float>(6, 5);
    test_norms<float>(5, 6);
    test_norms<double>(6, 5);
    test_norms<double>(5, 6);
}
