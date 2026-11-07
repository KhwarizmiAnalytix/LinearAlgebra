#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/determinant.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "test_backend.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t>
std::vector<value_t> matvec(const dense_matrix<value_t>& a, const std::vector<value_t>& x)
{
    std::vector<value_t> r(a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        value_t sum = 0;
        for (std::size_t j = 0; j < a.cols(); ++j)
        {
            sum += a(i, j) * x[j];
        }
        r[i] = sum;
    }
    return r;
}

template <typename value_t> value_t hmax_abs(const dense_matrix<value_t>& a)
{
    value_t max_error = 0;
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < a.cols(); ++j)
        {
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(a(i, j))));
        }
    }
    return max_error;
}

template <typename value_t>
value_t hmax_abs_diff(const dense_matrix<value_t>& a, const dense_matrix<value_t>& b)
{
    value_t max_error = 0;
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < a.cols(); ++j)
        {
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(a(i, j) - b(i, j))));
        }
    }
    return max_error;
}

template <typename value_t>
value_t hmax_abs_diff(const std::vector<value_t>& a, const std::vector<value_t>& b)
{
    value_t max_error = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        max_error = std::fmax(max_error, std::fabs(static_cast<double>(a[i] - b[i])));
    }
    return max_error;
}

template <typename value_t, linalg::linalg_backend Backend> void test_lu(double tolerance)
{
    const std::size_t n = 11;

    std::vector<linalg_int> pivot(n + 1);

    std::vector<value_t> b(n);

    dense_matrix<value_t> A(n, n);
    dense_matrix<value_t> a1(n, n);

    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            A(i, j) = static_cast<value_t>(std::fabs(rand() / static_cast<double>(RAND_MAX)));
        }
        // Diagonally dominant (off-diagonals are each < 1, so their row sum is < n - 1 < n):
        // keeps A's conditioning independent of the unseeded, shared rand() stream's position,
        // which otherwise shifts with how many random draws earlier tests in the binary made.
        A(i, i) = static_cast<value_t>(n);
        b[i]    = static_cast<value_t>(rand()) / static_cast<value_t>(RAND_MAX);
    }

    a1 = A;
    const auto lu_ws_size =
        linalg::lu_decomposition<value_t, Backend>::workspace_size(static_cast<linalg_int>(n));
    std::vector<unsigned char> lu_ws(lu_ws_size);
    linalg::lu_decomposition<value_t, Backend>::apply(
        a1.data(), static_cast<linalg_int>(n), pivot.data(), lu_ws.data(), lu_ws_size);

    dense_matrix<value_t> L(n, n);
    dense_matrix<value_t> U(n, n);

    for (std::size_t i = 0; i < n; i++)
    {
        U(i, i) = a1(i, i);
        L(i, i) = 1.;
        for (std::size_t j = 0; j < i; j++)
        {
            L(i, j) = a1(i, j);
            U(j, i) = a1(j, i);
        }
    }
    dense_matrix<value_t> result = matmul(L, U);

    // mkl/blas_lapacke always pivot (LAPACKE getrf); the scalar native path
    // only does when LINALG_HAS_LU_PIVOTING is on.
    constexpr bool kBackendPivots = Backend == linalg::linalg_backend::MKL ||
                                    Backend == linalg::linalg_backend::BLAS_LAPACKE ||
                                    LINALG_HAS_LU_PIVOTING;
    if (kBackendPivots)
    {
        for (linalg_int i = static_cast<linalg_int>(n) - 1; i >= 0; --i)
        {
            auto p = pivot[i] - 1;
            if (p != i)
            {
                for (linalg_int j = 0; j < static_cast<linalg_int>(n); ++j)
                {
                    auto ii        = static_cast<std::size_t>(i);
                    auto pp        = static_cast<std::size_t>(p);
                    auto jj        = static_cast<std::size_t>(j);
                    auto temp      = result(ii, jj);
                    result(ii, jj) = result(pp, jj);
                    result(pp, jj) = temp;
                }
            }
        }
    }

    auto diff = hmax_abs_diff(result, A);
    EXPECT_LE(diff, tolerance);

    dense_matrix<value_t> IA = a1;

    const auto invert_ws_size = linalg::inversion<value_t, Backend>::workspace_size(
        static_cast<linalg_int>(n), linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);
    std::vector<unsigned char> invert_ws(invert_ws_size);
    linalg::inversion<value_t, Backend>::apply(IA.data(),
        pivot.data(),
        static_cast<linalg_int>(n),
        linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER,
        invert_ws.data(),
        invert_ws_size);

    const auto det_ws_size =
        linalg::determinant<value_t, linalg::linalg_backend::NATIVE>::workspace_size(
            static_cast<linalg_int>(n), linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);
    std::vector<unsigned char> det_ws(det_ws_size);
    linalg::determinant<value_t, linalg::linalg_backend::NATIVE>::apply(a1.data(),
        pivot.data(),
        static_cast<linalg_int>(n),
        linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER,
        det_ws.data(),
        det_ws_size);

    dense_matrix<value_t> Id(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        Id(i, i) = 1.;
    }

    result = matmul(IA, A);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            result(i, j) -= Id(i, j);
        }
    }

    diff = hmax_abs(result);
    EXPECT_LE(diff, tolerance);

    std::vector<value_t> x             = b;
    const auto           solve_ws_size = linalg::linear_solver<value_t, Backend>::workspace_size(
        static_cast<linalg_int>(n), linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);
    std::vector<unsigned char> solve_ws(solve_ws_size);
    linalg::linear_solver<value_t, Backend>::apply(a1.data(),
        pivot.data(),
        static_cast<linalg_int>(n),
        x.data(),
        linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER,
        solve_ws.data(),
        solve_ws_size);

    diff = hmax_abs_diff(b, matvec(A, x));
    EXPECT_LE(diff, tolerance);
}

// Factors a copy of A and returns P * L * U rebuilt from the in-place
// result and the 1-based LAPACK-style pivots, plus the reported info.
template <typename value_t, linalg::linalg_backend Backend>
dense_matrix<value_t> lu_reconstruct(const dense_matrix<value_t>& A, int& info, bool& ok)
{
    const std::size_t       n  = A.rows();
    dense_matrix<value_t>   a1 = A;
    std::vector<linalg_int> pivot(n, -1);
    const auto              ws_size =
        linalg::lu_decomposition<value_t, Backend>::workspace_size(static_cast<linalg_int>(n));
    std::vector<unsigned char> ws(ws_size);
    info = -1;
    ok   = linalg::lu_decomposition<value_t, Backend>::apply(
        a1.data(), static_cast<linalg_int>(n), pivot.data(), ws.data(), ws_size, &info);

    dense_matrix<value_t> L(n, n);
    dense_matrix<value_t> U(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        L(i, i) = 1;
        for (std::size_t j = 0; j < n; ++j)
        {
            if (j < i)
            {
                L(i, j) = a1(i, j);
            }
            else
            {
                U(i, j) = a1(i, j);
            }
        }
    }
    dense_matrix<value_t> result = matmul(L, U);
    for (std::size_t i = n; i-- > 0;)
    {
        // Every backend, pivoting or not, must leave valid 1-based pivots.
        EXPECT_GE(pivot[i], static_cast<linalg_int>(i + 1));
        EXPECT_LE(pivot[i], static_cast<linalg_int>(n));
        const auto p = static_cast<std::size_t>(pivot[i] - 1);
        if (p != i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                std::swap(result(i, j), result(p, j));
            }
        }
    }
    return result;
}

template <typename value_t>
dense_matrix<value_t> make_matrix(std::size_t n, std::initializer_list<double> values)
{
    dense_matrix<value_t> A(n, n);
    auto                  it = values.begin();
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            A(i, j) = static_cast<value_t>(*it++);
        }
    }
    return A;
}

// A zero pivot must not stop the factorization (LAPACK getrf semantics):
// info names the first zero on U's diagonal and m/pivot still hold a full
// A = P * L * U. The zero pivots arise exactly (zero columns, or
// power-of-two eliminations); other multipliers may round, hence tol.
template <typename value_t, linalg::linalg_backend Backend> void test_lu_zero_pivot()
{
    int           info = 0;
    bool          ok   = true;
    const value_t tol  = 16 * std::numeric_limits<value_t>::epsilon();

    // Column 0 entirely zero: zero pivot at the very first step.
    const auto A0 = make_matrix<value_t>(3, {0, 1, 2, 0, 3, 4, 0, 5, 7});
    EXPECT_LE(hmax_abs_diff(lu_reconstruct<value_t, Backend>(A0, info, ok), A0), tol);
    EXPECT_EQ(info, 1);
    EXPECT_FALSE(ok);

    // Rank-deficient: column 1 becomes exactly zero after one elimination
    // step, with a later column still left to factor.
    const auto A1 = make_matrix<value_t>(3, {1, 2, 3, 2, 4, 5, 4, 8, 1});
    EXPECT_LE(hmax_abs_diff(lu_reconstruct<value_t, Backend>(A1, info, ok), A1), tol);
    EXPECT_EQ(info, 2);
    EXPECT_FALSE(ok);

    // All-zero matrix: the first zero pivot is reported, not the last.
    const auto Z = make_matrix<value_t>(2, {0, 0, 0, 0});
    EXPECT_LE(hmax_abs_diff(lu_reconstruct<value_t, Backend>(Z, info, ok), Z), tol);
    EXPECT_EQ(info, 1);
    EXPECT_FALSE(ok);
}

// Only an exactly-zero pivot is singular: a perfectly conditioned matrix at
// a tiny scale (formerly rejected by an absolute epsilon threshold in the
// native backend) and a subnormal pivot must both factor successfully.
template <typename value_t, linalg::linalg_backend Backend> void test_lu_small_scale()
{
    int  info = -1;
    bool ok   = false;

    const auto s  = static_cast<value_t>(1e-8);
    const auto SI = make_matrix<value_t>(2, {1e-8, 0, 0, 1e-8});
    const auto R  = lu_reconstruct<value_t, Backend>(SI, info, ok);
    EXPECT_EQ(info, 0);
    EXPECT_TRUE(ok);
    EXPECT_LE(hmax_abs_diff(R, SI), s * std::numeric_limits<value_t>::epsilon());

    // [[t, 1], [t/2, 1]] with t subnormal: 1/t overflows, so the multiplier
    // must come from a division. Skipped where subnormals are flushed.
    const value_t t = std::ldexp(value_t(1), std::numeric_limits<value_t>::min_exponent - 4);
    if (t != value_t(0) && t < std::numeric_limits<value_t>::min())
    {
        dense_matrix<value_t> T2(2, 2);
        T2(0, 0)      = t;
        T2(0, 1)      = 1;
        T2(1, 0)      = t / 2;
        T2(1, 1)      = 1;
        const auto RT = lu_reconstruct<value_t, Backend>(T2, info, ok);
        EXPECT_EQ(info, 0);
        EXPECT_TRUE(ok);
        EXPECT_EQ(hmax_abs_diff(RT, T2), value_t(0));
    }
}
}  // namespace

TEST(Math, LUDecompositionZeroPivot)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr linalg::linalg_backend Backend = decltype(backend)::value;
            test_lu_zero_pivot<float, Backend>();
            test_lu_zero_pivot<double, Backend>();
            test_lu_small_scale<float, Backend>();
            test_lu_small_scale<double, Backend>();
        });
}

TEST(Math, LUDecomposition)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr linalg::linalg_backend Backend = decltype(backend)::value;
            test_lu<float, Backend>(2.e-5);
            test_lu<double, Backend>(4.e-14);
        });
}
