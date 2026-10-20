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

template <typename> struct NormTolerance
{
};
template <> struct NormTolerance<float>
{
    static constexpr double value = 1e-4;
};
template <> struct NormTolerance<double>
{
    static constexpr double value = 1e-12;
};

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
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::FROBENIUS,
            nullptr,
            0);
    EXPECT_NEAR(static_cast<double>(frob), frob_expected, NormTolerance<value_t>::value);

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
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::ONE,
            nullptr,
            0);
    EXPECT_NEAR(static_cast<double>(one), one_expected, NormTolerance<value_t>::value);

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
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::INFINITY_NORM,
            nullptr,
            0);
    EXPECT_NEAR(static_cast<double>(inf), inf_expected, NormTolerance<value_t>::value);

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
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::workspace_size(
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            linalg::norm_type::TWO);
    std::vector<unsigned char> two_ws(two_ws_size);
    const auto                 two =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::TWO,
            two_ws.data(),
            two_ws_size);
    EXPECT_NEAR(static_cast<double>(two), static_cast<double>(smax), NormTolerance<value_t>::value);
}

// Zero matrix: every norm must be exactly 0.
// Mirrors torch.linalg.norm on a zero tensor (PyTorch test_norm_zero_matrix).
template <typename value_t>
void test_norm_zero_matrix(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> Z(rows, columns);
    Z.fill(value_t(0));

    for (auto type : {linalg::norm_type::FROBENIUS,
             linalg::norm_type::ONE,
             linalg::norm_type::INFINITY_NORM})
    {
        const auto n =
            linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(Z.begin(),
                static_cast<linalg_long>(rows),
                static_cast<linalg_long>(columns),
                static_cast<linalg_long>(columns),
                type,
                nullptr,
                0);
        EXPECT_EQ(static_cast<double>(n), 0.0);
    }
}

// Identity matrix: known exact norm values for n x n I.
//   Frobenius = sqrt(n),  1-norm = 1,  inf-norm = 1,  2-norm = 1.
// Mirrors torch.linalg.norm on torch.eye (PyTorch test_norm_known_matrix).
template <typename value_t>
void test_norm_identity(std::size_t n)
{
    dense_matrix<value_t> I(n, n);
    I.fill(value_t(0));
    for (std::size_t i = 0; i < n; i++)
    {
        I(i, i) = value_t(1);
    }

    const auto frob =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(I.begin(),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            linalg::norm_type::FROBENIUS,
            nullptr,
            0);
    EXPECT_NEAR(static_cast<double>(frob), std::sqrt(static_cast<double>(n)), NormTolerance<value_t>::value);

    const auto one =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(I.begin(),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            linalg::norm_type::ONE,
            nullptr,
            0);
    EXPECT_NEAR(static_cast<double>(one), 1.0, NormTolerance<value_t>::value);

    const auto inf =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(I.begin(),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            linalg::norm_type::INFINITY_NORM,
            nullptr,
            0);
    EXPECT_NEAR(static_cast<double>(inf), 1.0, NormTolerance<value_t>::value);

    const auto two_ws_size =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::workspace_size(
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            linalg::norm_type::TWO);
    std::vector<unsigned char> two_ws(two_ws_size);
    const auto                 two =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(I.begin(),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            linalg::norm_type::TWO,
            two_ws.data(),
            two_ws_size);
    EXPECT_NEAR(static_cast<double>(two), 1.0, NormTolerance<value_t>::value);
}

// Norm inequality: ||A||_2 <= ||A||_F <= sqrt(min(m,n)) * ||A||_2.
// This is a classical bound that PyTorch implicitly tests via norm consistency
// checks (PyTorch test_norm_inequality).
template <typename value_t, linalg::linalg_backend Backend>
void test_norm_inequality(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> M(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            M(i, j) = static_cast<value_t>(20. * rand() / RAND_MAX - 10.);
        }
    }

    const auto frob =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(M.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::FROBENIUS,
            nullptr,
            0);

    dense_matrix<value_t> M_copy(rows, columns);
    for (std::size_t i = 0; i < rows * columns; i++)
    {
        M_copy.begin()[i] = M.begin()[i];
    }
    const auto two_ws_size =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::workspace_size(
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            linalg::norm_type::TWO);
    std::vector<unsigned char> two_ws(two_ws_size);
    const auto                 two =
        linalg::norm<value_t, linalg::linalg_backend::NATIVE>::apply(M_copy.begin(),
            static_cast<linalg_long>(rows),
            static_cast<linalg_long>(columns),
            static_cast<linalg_long>(columns),
            linalg::norm_type::TWO,
            two_ws.data(),
            two_ws_size);

    const double f = static_cast<double>(frob);
    const double s = static_cast<double>(two);
    const double k = static_cast<double>(std::min(rows, columns));

    // ||A||_2 <= ||A||_F (with a small numerical margin)
    EXPECT_LE(s, f + NormTolerance<value_t>::value * std::fabs(f));
    // ||A||_F <= sqrt(k) * ||A||_2
    EXPECT_LE(f, std::sqrt(k) * s + NormTolerance<value_t>::value * std::fabs(s));
}
}  // namespace

TEST(Math, MatrixNorm)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_norms<float, Backend>(6, 5);
        test_norms<float, Backend>(5, 6);
        test_norms<double, Backend>(6, 5);
        test_norms<double, Backend>(5, 6);
    });
}

TEST(Math, MatrixNormZeroMatrix)
{
    test_norm_zero_matrix<float>(5, 4);
    test_norm_zero_matrix<float>(4, 5);
    test_norm_zero_matrix<double>(5, 4);
    test_norm_zero_matrix<double>(4, 5);
}

TEST(Math, MatrixNormIdentity)
{
    test_norm_identity<float>(1);
    test_norm_identity<float>(4);
    test_norm_identity<double>(1);
    test_norm_identity<double>(4);
}

TEST(Math, MatrixNormInequality)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_norm_inequality<float, Backend>(6, 4);
        test_norm_inequality<float, Backend>(4, 6);
        test_norm_inequality<double, Backend>(6, 4);
        test_norm_inequality<double, Backend>(4, 6);
    });
}
