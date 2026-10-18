#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/rank.h"
#include "include/matrix_operation/condition_number.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct ConditionTolerance
{
};
template <> struct ConditionTolerance<float>
{
    static constexpr double value = 1e-3;
};
template <> struct ConditionTolerance<double>
{
    static constexpr double value = 1e-10;
};

template <typename T, linalg::linalg_backend Backend>
linalg_long compute_rank(const dense_matrix<T>& A, T tol)
{
    const auto ws_size = linalg::rank<T, Backend>::workspace_size(
        static_cast<linalg_long>(A.rows()), static_cast<linalg_long>(A.cols()));
    std::vector<unsigned char> ws(ws_size);
    return linalg::rank<T, Backend>::apply(A.begin(),
        static_cast<linalg_long>(A.rows()),
        static_cast<linalg_long>(A.cols()),
        static_cast<linalg_long>(A.cols()),
        tol,
        ws.data(),
        ws_size);
}

// Diagonal n×n matrix: singular values are exactly the diagonal entries.
template <typename T>
dense_matrix<T> diagonal(const std::vector<double>& sigma)
{
    const std::size_t n = sigma.size();
    dense_matrix<T>   A(n, n);
    A.fill(T(0));
    for (std::size_t i = 0; i < n; i++)
        A(i, i) = static_cast<T>(sigma[i]);
    return A;
}

template <typename T, linalg::linalg_backend Backend>
void test_rank_full(std::size_t rows, std::size_t columns)
{
    dense_matrix<T> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
        for (std::size_t j = 0; j < columns; j++)
            A(i, j) = static_cast<T>(20. * rand() / RAND_MAX - 10.);

    EXPECT_EQ((compute_rank<T, Backend>(A, T(-1))), static_cast<linalg_long>(std::min(rows, columns)));
}

// Rank-1 matrix built as an outer product: u * v^T always has rank <= 1.
template <typename T, linalg::linalg_backend Backend>
void test_rank_deficient(std::size_t n)
{
    dense_matrix<T> u(n, 1);
    dense_matrix<T> v(n, 1);
    for (std::size_t i = 0; i < n; i++)
    {
        u(i, 0) = static_cast<T>(1. + rand() / static_cast<double>(RAND_MAX));
        v(i, 0) = static_cast<T>(1. + rand() / static_cast<double>(RAND_MAX));
    }
    dense_matrix<T> A(n, n);
    for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++)
            A(i, j) = u(i, 0) * v(j, 0);

    EXPECT_EQ((compute_rank<T, Backend>(A, T(-1))), static_cast<linalg_long>(1));
}

// Zero matrix: no singular value exceeds any positive cutoff, rank = 0.
template <typename T, linalg::linalg_backend Backend>
void test_rank_zero_matrix(std::size_t rows, std::size_t cols)
{
    dense_matrix<T> Z(rows, cols);
    Z.fill(T(0));
    EXPECT_EQ((compute_rank<T, Backend>(Z, T(-1))), static_cast<linalg_long>(0));
}

// Diagonal matrices with prescribed singular values and an explicit tolerance.
// Mirrors numpy.linalg.matrix_rank(A, tol=...) boundary tests.
// For a diagonal matrix the singular values are the diagonal entries exactly,
// so the expected rank is the count of entries strictly greater than tol.
template <typename T, linalg::linalg_backend Backend>
void test_rank_known_tolerance()
{
    // sigma = [10, 1, 0.1, 0.01]:
    //   tol=0.05  → rank 3  (10, 1, 0.1 > 0.05; 0.01 < 0.05)
    //   tol=0.5   → rank 2  (10, 1 > 0.5;  0.1, 0.01 < 0.5)
    //   tol=5     → rank 1  (10 > 5; rest < 5)
    const auto A = diagonal<T>({10.0, 1.0, 0.1, 0.01});
    EXPECT_EQ((compute_rank<T, Backend>(A, T(0.05))),  static_cast<linalg_long>(3));
    EXPECT_EQ((compute_rank<T, Backend>(A, T(0.5))),   static_cast<linalg_long>(2));
    EXPECT_EQ((compute_rank<T, Backend>(A, T(5.0))),   static_cast<linalg_long>(1));
}

// Boundary straddle: one singular value sits just above the cutoff, one just below.
// This is the most sensitive correctness test for the threshold comparison.
template <typename T, linalg::linalg_backend Backend>
void test_rank_boundary_straddle()
{
    // sigma = [1.0, 2*tol, 0.5*tol, tiny]; tol = 1e-3
    // Expected: values 1.0 and 2e-3 count; 5e-4 and tiny do not → rank 2.
    constexpr double tol_val = 1e-3;
    const auto       A       = diagonal<T>({1.0, 2.0 * tol_val, 0.5 * tol_val, 1e-9});
    EXPECT_EQ((compute_rank<T, Backend>(A, static_cast<T>(tol_val))), static_cast<linalg_long>(2));

    // Shift the cutoff so that 2*tol is now also excluded → rank 1.
    EXPECT_EQ((compute_rank<T, Backend>(A, static_cast<T>(3.0 * tol_val))), static_cast<linalg_long>(1));
}

template <typename T, linalg::linalg_backend Backend>
double compute_condition_number(const dense_matrix<T>& A)
{
    const auto ws_size = linalg::condition_number<T, Backend>::workspace_size(
        static_cast<linalg_long>(A.rows()), static_cast<linalg_long>(A.cols()));
    std::vector<unsigned char> ws(ws_size);
    return static_cast<double>(
        linalg::condition_number<T, Backend>::apply(A.begin(),
            static_cast<linalg_long>(A.rows()),
            static_cast<linalg_long>(A.cols()),
            static_cast<linalg_long>(A.cols()),
            ws.data(),
            ws_size));
}

template <typename value_t, linalg::linalg_backend Backend>
void test_condition_number_identity(std::size_t n)
{
    dense_matrix<value_t> I(n, n);
    for (std::size_t i = 0; i < n; i++)
        I(i, i) = value_t(1);
    EXPECT_NEAR((compute_condition_number<value_t, Backend>(I)), 1.0, ConditionTolerance<value_t>::value);
}

// Diagonal matrices with known σ_max / σ_min ratios.
// Mirrors numpy.linalg.cond on diagonal inputs.
template <typename T, linalg::linalg_backend Backend>
void test_condition_number_known()
{
    const double tol = ConditionTolerance<T>::value;
    // diag(10, 2) → cond = 5
    EXPECT_NEAR((compute_condition_number<T, Backend>(diagonal<T>({10.0, 2.0}))), 5.0, tol);
    // diag(100, 4) → cond = 25
    EXPECT_NEAR((compute_condition_number<T, Backend>(diagonal<T>({100.0, 4.0}))), 25.0, tol);
    // diag(1, 0.01, 0.01) → cond = 100
    EXPECT_NEAR((compute_condition_number<T, Backend>(diagonal<T>({1.0, 0.01, 0.01}))), 100.0, tol);
}

// Singular matrix: σ_min = 0, condition number must be +infinity.
template <typename T, linalg::linalg_backend Backend>
void test_condition_number_singular()
{
    // diag(1, 0) — one zero singular value
    const auto cond = compute_condition_number<T, Backend>(diagonal<T>({1.0, 0.0}));
    EXPECT_TRUE(std::isinf(cond));
    // Full zero matrix — all singular values are zero
    dense_matrix<T> Z(3, 3);
    Z.fill(T(0));
    EXPECT_TRUE(std::isinf((compute_condition_number<T, Backend>(Z))));
}

}  // namespace

TEST(Math, MatrixRank)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_rank_full<float, Backend>(6, 4);
        test_rank_full<double, Backend>(6, 4);
        test_rank_deficient<float, Backend>(5);
        test_rank_deficient<double, Backend>(5);
    });
}

TEST(Math, MatrixRankZero)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_rank_zero_matrix<float, Backend>(4, 4);
        test_rank_zero_matrix<double, Backend>(4, 4);
        test_rank_zero_matrix<float, Backend>(6, 3);
        test_rank_zero_matrix<double, Backend>(6, 3);
    });
}

TEST(Math, MatrixRankKnownTolerance)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_rank_known_tolerance<float, Backend>();
        test_rank_known_tolerance<double, Backend>();
    });
}

TEST(Math, MatrixRankBoundaryStraddle)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_rank_boundary_straddle<float, Backend>();
        test_rank_boundary_straddle<double, Backend>();
    });
}

TEST(Math, MatrixConditionNumber)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_condition_number_identity<float, Backend>(5);
        test_condition_number_identity<double, Backend>(5);
    });
}

TEST(Math, MatrixConditionNumberKnown)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_condition_number_known<float, Backend>();
        test_condition_number_known<double, Backend>();
    });
}

TEST(Math, MatrixConditionNumberSingular)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_condition_number_singular<float, Backend>();
        test_condition_number_singular<double, Backend>();
    });
}
