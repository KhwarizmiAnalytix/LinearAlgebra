#include <cmath>
#include <cstdlib>
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
}  // namespace

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
