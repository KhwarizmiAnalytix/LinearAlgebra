#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/exponential.h"
#include "include/matrix_operation/multiplication.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct ExpTolerance
{
};

template <> struct ExpTolerance<float>
{
    static constexpr double tolerance = 5.e-4;
};

template <> struct ExpTolerance<double>
{
    static constexpr double tolerance = 1.e-9;
};

template <typename value_t> std::vector<unsigned char> exp_workspace(linalg_int n)
{
    const auto bytes =
        linalg::exponential<value_t, linalg::linalg_backend::native>::workspace_size(n);
    return std::vector<unsigned char>(bytes);
}

template <typename value_t> std::vector<unsigned char> gemm_workspace(linalg_int n)
{
    const auto bytes =
        linalg::multiplication<value_t, linalg::linalg_backend::native>::workspace_size(n, n, n);
    return std::vector<unsigned char>(bytes);
}

template <typename value_t>
double max_abs_diff(const dense_matrix<value_t>& a, const dense_matrix<value_t>& b, std::size_t n)
{
    double worst = 0.;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            worst = std::max(
                worst, std::fabs(static_cast<double>(a(i, j)) - static_cast<double>(b(i, j))));
        }
    }
    return worst;
}

// exp(0) == I.
template <typename value_t> void test_zero()
{
    const std::size_t     n = 4;
    dense_matrix<value_t> A(n, n);
    dense_matrix<value_t> R(n, n);
    auto                  ws = exp_workspace<value_t>(static_cast<linalg_int>(n));
    linalg::exponential<value_t, linalg::linalg_backend::native>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        R.begin(),
        static_cast<linalg_int>(n),
        ws.data(),
        ws.size());

    dense_matrix<value_t> I(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        I(i, i) = value_t(1);
    }
    EXPECT_LT(max_abs_diff(R, I, n), ExpTolerance<value_t>::tolerance);
}

// A diagonal: exp(A) is diagonal with exp of each entry.
template <typename value_t> void test_diagonal()
{
    const std::size_t     n = 3;
    dense_matrix<value_t> A(n, n);
    A(0, 0) = value_t(0.5);
    A(1, 1) = value_t(-1.25);
    A(2, 2) = value_t(2.0);

    dense_matrix<value_t> R(n, n);
    auto                  ws = exp_workspace<value_t>(static_cast<linalg_int>(n));
    linalg::exponential<value_t, linalg::linalg_backend::native>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        R.begin(),
        static_cast<linalg_int>(n),
        ws.data(),
        ws.size());

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const double expected = (i == j) ? std::exp(static_cast<double>(A(i, i))) : 0.;
            EXPECT_NEAR(static_cast<double>(R(i, j)), expected, ExpTolerance<value_t>::tolerance);
        }
    }
}

// Nilpotent N = [[0,1],[0,0]] (N^2 = 0): exp(N) = I + N = [[1,1],[0,1]].
template <typename value_t> void test_nilpotent()
{
    dense_matrix<value_t> A(2, 2);
    A(0, 1) = value_t(1);

    dense_matrix<value_t> R(2, 2);
    auto                  ws = exp_workspace<value_t>(2);
    linalg::exponential<value_t, linalg::linalg_backend::native>::apply(
        A.begin(), 2, 2, R.begin(), 2, ws.data(), ws.size());

    EXPECT_NEAR(static_cast<double>(R(0, 0)), 1., ExpTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(R(0, 1)), 1., ExpTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(R(1, 0)), 0., ExpTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(R(1, 1)), 1., ExpTolerance<value_t>::tolerance);
}

// Skew-symmetric rotation generator [[0,-t],[t,0]]: exp gives the 2D
// rotation matrix [[cos t, -sin t], [sin t, cos t]]. A large theta gives
// ||A||_1 well above the degree-13 threshold, exercising the scaling and
// repeated-squaring loop rather than just a direct Pade evaluation.
template <typename value_t> void test_rotation_generator(double theta)
{
    dense_matrix<value_t> A(2, 2);
    A(0, 1) = static_cast<value_t>(-theta);
    A(1, 0) = static_cast<value_t>(theta);

    dense_matrix<value_t> R(2, 2);
    auto                  ws = exp_workspace<value_t>(2);
    linalg::exponential<value_t, linalg::linalg_backend::native>::apply(
        A.begin(), 2, 2, R.begin(), 2, ws.data(), ws.size());

    EXPECT_NEAR(static_cast<double>(R(0, 0)), std::cos(theta), ExpTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(R(0, 1)), -std::sin(theta), ExpTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(R(1, 0)), std::sin(theta), ExpTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(R(1, 1)), std::cos(theta), ExpTolerance<value_t>::tolerance);
}

// exp(A) computed against a directly-summed, high-order Taylor series
// (independent of the scaling-and-squaring/Pade implementation under test)
// for a general, moderate-norm random matrix.
template <typename value_t> void test_against_taylor_series(std::size_t n, double scale = 1.0)
{
    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            // Keep ||A||_1 modest so the truncated series below converges
            // to double-precision accuracy well within kTaylorTerms. `scale`
            // lets callers target a specific ||A||_1 range to exercise a
            // particular degree bracket (T1/T2/T4/T8/T12/T18) of the
            // Bader-Blanes-Casas algorithm under test.
            A(i, j) = static_cast<value_t>(scale * (1. * rand() / RAND_MAX - 0.5));
        }
    }

    dense_matrix<double> term(n, n);
    dense_matrix<double> sum(n, n);
    dense_matrix<double> nextTerm(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        term(i, i) = 1.;
        sum(i, i)  = 1.;
    }

    dense_matrix<double> Ad(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            Ad(i, j) = static_cast<double>(A(i, j));
        }
    }

    constexpr int kTaylorTerms = 40;
    for (int k = 1; k <= kTaylorTerms; ++k)
    {
        auto gemm_ws = gemm_workspace<double>(static_cast<linalg_int>(n));
        linalg::multiplication<double, linalg::linalg_backend::native>::apply(false,
            false,
            static_cast<linalg_int>(n),
            static_cast<linalg_int>(n),
            static_cast<linalg_int>(n),
            term.begin(),
            static_cast<linalg_int>(n),
            Ad.begin(),
            static_cast<linalg_int>(n),
            nextTerm.begin(),
            static_cast<linalg_int>(n),
            gemm_ws.data(),
            gemm_ws.size());
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                nextTerm(i, j) /= static_cast<double>(k);
                sum(i, j) += nextTerm(i, j);
            }
        }
        term.fill(0.);
        std::swap(term, nextTerm);
    }

    dense_matrix<value_t> R(n, n);
    auto                  ws = exp_workspace<value_t>(static_cast<linalg_int>(n));
    linalg::exponential<value_t, linalg::linalg_backend::native>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        R.begin(),
        static_cast<linalg_int>(n),
        ws.data(),
        ws.size());

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            EXPECT_NEAR(static_cast<double>(R(i, j)), sum(i, j), ExpTolerance<value_t>::tolerance);
        }
    }
}

// exp(A) * exp(-A) == I, a property independent of any closed-form value.
template <typename value_t> void test_inverse_property(std::size_t n)
{
    dense_matrix<value_t> A(n, n);
    dense_matrix<value_t> negA(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            A(i, j)    = static_cast<value_t>((1. * rand() / RAND_MAX - 0.5));
            negA(i, j) = -A(i, j);
        }
    }

    dense_matrix<value_t> expA(n, n);
    dense_matrix<value_t> expNegA(n, n);
    auto                  ws = exp_workspace<value_t>(static_cast<linalg_int>(n));
    linalg::exponential<value_t, linalg::linalg_backend::native>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        expA.begin(),
        static_cast<linalg_int>(n),
        ws.data(),
        ws.size());
    linalg::exponential<value_t, linalg::linalg_backend::native>::apply(negA.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        expNegA.begin(),
        static_cast<linalg_int>(n),
        ws.data(),
        ws.size());

    dense_matrix<value_t> product(n, n);
    auto                  gemm_ws = gemm_workspace<value_t>(static_cast<linalg_int>(n));
    linalg::multiplication<value_t, linalg::linalg_backend::native>::apply(false,
        false,
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        expA.begin(),
        static_cast<linalg_int>(n),
        expNegA.begin(),
        static_cast<linalg_int>(n),
        product.begin(),
        static_cast<linalg_int>(n),
        gemm_ws.data(),
        gemm_ws.size());

    dense_matrix<value_t> I(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        I(i, i) = value_t(1);
    }
    EXPECT_LT(max_abs_diff(product, I, n), ExpTolerance<value_t>::tolerance);
}
}  // namespace

TEST(Math, MatrixExponential)
{
    test_zero<float>();
    test_zero<double>();

    test_diagonal<float>();
    test_diagonal<double>();

    test_nilpotent<float>();
    test_nilpotent<double>();

    test_rotation_generator<float>(0.7);
    test_rotation_generator<double>(0.7);
    test_rotation_generator<float>(20.0);
    test_rotation_generator<double>(20.0);

    test_against_taylor_series<float>(5);
    test_against_taylor_series<double>(6);

    // Double theta thresholds (degree: upper bound on ||A||_1) are
    // T1: 2.22e-16, T2: 2.58e-8, T4: 3.40e-4, T8: 4.99e-2, T12: 3.00e-1,
    // T18 (scale-and-square beyond): 1.09e0. These magnitudes keep ||A||_1
    // comfortably inside each bracket in turn, so every degree path
    // (T2/T4/T8/T12 in particular, not otherwise exercised above) is
    // checked against the independent Taylor-series reference.
    test_against_taylor_series<double>(3, 1e-8);    // T2 bracket
    test_against_taylor_series<double>(3, 1e-4);    // T4 bracket
    test_against_taylor_series<double>(3, 1e-2);    // T8 bracket
    test_against_taylor_series<double>(3, 1.5e-1);  // T12 bracket
    test_against_taylor_series<float>(3, 1e-8);     // T1 bracket (float)
    test_against_taylor_series<float>(3, 3e-4);     // T2 bracket (float)
    test_against_taylor_series<float>(3, 1e-2);     // T4 bracket (float)
    test_against_taylor_series<float>(3, 3e-1);     // T8 bracket (float)

    test_inverse_property<float>(5);
    test_inverse_property<double>(6);
}
