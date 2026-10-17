#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/power.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct PowerTolerance
{
};

template <> struct PowerTolerance<float>
{
    static constexpr double tolerance = 5.e-4;
};

template <> struct PowerTolerance<double>
{
    static constexpr double tolerance = 1.e-9;
};

template <typename value_t> std::vector<unsigned char> power_workspace(linalg_int n)
{
    const auto bytes = linalg::power<value_t, linalg::linalg_backend::native>::workspace_size(n);
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

template <typename value_t> double max_abs(const dense_matrix<value_t>& a, std::size_t n)
{
    double worst = 0.;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            worst = std::max(worst, std::fabs(static_cast<double>(a(i, j))));
        }
    }
    return worst;
}

template <typename value_t> dense_matrix<value_t> identity(std::size_t n)
{
    dense_matrix<value_t> I(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        I(i, i) = value_t(1);
    }
    return I;
}

template <typename value_t>
dense_matrix<value_t> apply_power(const dense_matrix<value_t>& A, std::size_t n, linalg_int e)
{
    dense_matrix<value_t> R(n, n);
    auto                  ws = power_workspace<value_t>(static_cast<linalg_int>(n));
    linalg::power<value_t, linalg::linalg_backend::native>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        e,
        R.begin(),
        static_cast<linalg_int>(n),
        ws.data(),
        ws.size());
    return R;
}

template <typename value_t>
dense_matrix<value_t> matmul(
    const dense_matrix<value_t>& a, const dense_matrix<value_t>& b, std::size_t n)
{
    dense_matrix<value_t> c(n, n);
    auto                  gemm_ws = gemm_workspace<value_t>(static_cast<linalg_int>(n));
    linalg::multiplication<value_t, linalg::linalg_backend::native>::apply(false,
        false,
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        a.begin(),
        static_cast<linalg_int>(n),
        b.begin(),
        static_cast<linalg_int>(n),
        c.begin(),
        static_cast<linalg_int>(n),
        gemm_ws.data(),
        gemm_ws.size());
    return c;
}

template <typename value_t> dense_matrix<value_t> random_matrix(std::size_t n)
{
    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            // Diagonally dominant so the matrix (and the powers/inverses exercised below)
            // stay well conditioned.
            A(i, j) = static_cast<value_t>(1. * rand() / RAND_MAX - 0.5);
        }
        A(i, i) += value_t(static_cast<value_t>(n));
    }
    return A;
}

// A^0 == I, regardless of A.
template <typename value_t> void test_zero_power(std::size_t n)
{
    const auto A = random_matrix<value_t>(n);
    const auto R = apply_power(A, n, 0);
    EXPECT_LT(max_abs_diff(R, identity<value_t>(n), n), PowerTolerance<value_t>::tolerance);
}

// A^1 == A.
template <typename value_t> void test_identity_power(std::size_t n)
{
    const auto A = random_matrix<value_t>(n);
    const auto R = apply_power(A, n, 1);
    EXPECT_LT(max_abs_diff(R, A, n), PowerTolerance<value_t>::tolerance);
}

// A^2 and A^3 against direct multiplication.
template <typename value_t> void test_small_positive_powers(std::size_t n)
{
    const auto A  = random_matrix<value_t>(n);
    const auto A2 = matmul(A, A, n);
    const auto A3 = matmul(A2, A, n);

    EXPECT_LT(max_abs_diff(apply_power(A, n, 2), A2, n), PowerTolerance<value_t>::tolerance);
    EXPECT_LT(max_abs_diff(apply_power(A, n, 3), A3, n), PowerTolerance<value_t>::tolerance);
}

// A^k for a larger k, built by repeated multiplication, exercises the binary
// (square-and-multiply) exponentiation path with more than one set bit.
template <typename value_t> void test_large_positive_power(std::size_t n, linalg_int k)
{
    const auto            A         = random_matrix<value_t>(n);
    dense_matrix<value_t> reference = identity<value_t>(n);
    for (linalg_int i = 0; i < k; ++i)
    {
        reference = matmul(reference, A, n);
    }

    const double scale = std::max(1.0, max_abs(reference, n));
    EXPECT_LT(max_abs_diff(apply_power(A, n, k), reference, n),
        PowerTolerance<value_t>::tolerance * scale);
}

// A^-1 matches an explicit inverse computed via A^-1 * A == I.
template <typename value_t> void test_negative_one_power(std::size_t n)
{
    const auto A    = random_matrix<value_t>(n);
    const auto Ainv = apply_power(A, n, -1);
    const auto prod = matmul(Ainv, A, n);
    EXPECT_LT(max_abs_diff(prod, identity<value_t>(n), n), PowerTolerance<value_t>::tolerance);
}

// A^-k == (A^-1)^k.
template <typename value_t> void test_negative_power(std::size_t n, linalg_int k)
{
    const auto A         = random_matrix<value_t>(n);
    const auto Ainv      = apply_power(A, n, -1);
    const auto reference = apply_power(Ainv, n, k);
    const auto R         = apply_power(A, n, -k);
    EXPECT_LT(max_abs_diff(R, reference, n), PowerTolerance<value_t>::tolerance);
}

}  // namespace

TEST(Math, MatrixPower)
{
    test_zero_power<float>(4);
    test_zero_power<double>(5);

    test_identity_power<float>(4);
    test_identity_power<double>(5);

    test_small_positive_powers<float>(4);
    test_small_positive_powers<double>(5);

    test_large_positive_power<float>(4, 7);
    test_large_positive_power<double>(5, 11);

    test_negative_one_power<float>(4);
    test_negative_one_power<double>(5);

    test_negative_power<float>(4, 5);
    test_negative_power<double>(5, 6);
}
