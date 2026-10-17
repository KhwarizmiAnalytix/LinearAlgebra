#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/norm.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t, linalg::linalg_backend Backend>
void test_norms(std::size_t rows, std::size_t columns)
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

    const auto frob =
        linalg::norm<value_t, linalg::linalg_backend::native>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::FROBENIUS,
            nullptr,
            0);
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
    const auto one =
        linalg::norm<value_t, linalg::linalg_backend::native>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::ONE,
            nullptr,
            0);
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
    const auto inf =
        linalg::norm<value_t, linalg::linalg_backend::native>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::INFINITY_NORM,
            nullptr,
            0);
    EXPECT_NEAR(static_cast<double>(inf), inf_expected, 1e-3);

    // Two-norm: cross-check against the largest singular value from
    // svd_decomposition directly.
    dense_matrix<value_t> A_copy(rows, columns);
    for (std::size_t i = 0; i < rows * columns; i++)
    {
        A_copy.begin()[i] = M.begin()[i];
    }
    std::size_t           k = std::min(rows, columns);
    std::vector<value_t>  S(k);
    dense_matrix<value_t> U(rows, k);
    dense_matrix<value_t> VT(k, columns);
    const auto svd_ws_size = linalg::svd_decomposition<value_t, Backend>::workspace_size(
        static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> svd_ws(svd_ws_size);
    linalg::svd_decomposition<value_t, Backend>::apply(static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        A_copy.begin(),
        static_cast<linalg_long>(columns),
        S.data(),
        U.begin(),
        static_cast<linalg_long>(k),
        VT.begin(),
        static_cast<linalg_long>(columns),
        svd_ws.data(),
        svd_ws_size);
    value_t smax = 0;
    for (auto s : S)
    {
        smax = std::max(smax, static_cast<value_t>(std::fabs(s)));
    }

    const auto two_ws_size =
        linalg::norm<value_t, linalg::linalg_backend::native>::workspace_size(
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            linalg::norm_type::TWO);
    std::vector<unsigned char> two_ws(two_ws_size);
    const auto                 two =
        linalg::norm<value_t, linalg::linalg_backend::native>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::TWO,
            two_ws.data(),
            two_ws_size);
    EXPECT_NEAR(static_cast<double>(two), static_cast<double>(smax), 1e-3);
}
}  // namespace

TEST(Math, MatrixNorm)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr auto Backend = decltype(backend)::value;
        test_norms<float, Backend>(6, 5);
        test_norms<float, Backend>(5, 6);
        test_norms<double, Backend>(6, 5);
        test_norms<double, Backend>(5, 6);
    });
}
