#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "linear_algebra/matrix_operation/linear_solver.h"
#include "linear_algebra/matrix_operation/lu_decomposition.h"
#include "linear_algebra/matrix_operation/matrix_inversion.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t>
dense_matrix<value_t> matmul(const dense_matrix<value_t>& a, const dense_matrix<value_t>& b)
{
    dense_matrix<value_t> c(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < b.cols(); ++j)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < a.cols(); ++k)
            {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
    return c;
}

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

template <typename value_t>
value_t hmax_abs(const dense_matrix<value_t>& a)
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

template <typename value_t>
void test_lu(double tolerance)
{
    const std::size_t n = 11;

    std::vector<quarisma_int> pivot(n + 1);

    std::vector<value_t> b(n);

    dense_matrix<value_t> A(n, n);
    dense_matrix<value_t> a1(n, n);

    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            A(i, j) = static_cast<value_t>(std::fabs(rand() / static_cast<double>(RAND_MAX)));
        }
        A(i, i) = 1.;
        b[i]    = static_cast<value_t>(rand()) / static_cast<value_t>(RAND_MAX);
    }

    a1 = A;
    linalg::lu_decomposition(a1.data(), static_cast<quarisma_int>(n), pivot.data());

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

#if defined(LINALG_LU_PIVOTING) || defined(LINALG_ENABLE_MKL)
    for (quarisma_int i = static_cast<quarisma_int>(n) - 1; i >= 0; --i)
    {
        auto p = pivot[i] - 1;
        if (p != i)
        {
            for (quarisma_int j = 0; j < static_cast<quarisma_int>(n); ++j)
            {
                auto ii    = static_cast<std::size_t>(i);
                auto pp    = static_cast<std::size_t>(p);
                auto jj    = static_cast<std::size_t>(j);
                auto temp  = result(ii, jj);
                result(ii, jj) = result(pp, jj);
                result(pp, jj) = temp;
            }
        }
    }
#endif

    auto diff = hmax_abs_diff(result, A);
    EXPECT_LE(diff, tolerance);

    dense_matrix<value_t> IA = a1;

    linalg::matrix_invert(
        IA.data(), pivot.data(), static_cast<quarisma_int>(n), linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);
    linalg::matrix_determinant(
        a1.data(), pivot.data(), static_cast<quarisma_int>(n), linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);

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

    std::vector<value_t> x = b;
    linalg::linear_solver(
        a1.data(),
        pivot.data(),
        static_cast<quarisma_int>(n),
        x.data(),
        linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);

    diff = hmax_abs_diff(b, matvec(A, x));
    EXPECT_LE(diff, tolerance);
}
}  // namespace

TEST(Math, LUDecomposition)
{
    test_lu<float>(2.e-5);
    test_lu<double>(4.e-14);
}
