#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/linear_solver.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t>
dense_matrix<value_t> transpose(const dense_matrix<value_t>& a)
{
    dense_matrix<value_t> t(a.cols(), a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < a.cols(); ++j)
        {
            t(j, i) = a(i, j);
        }
    }
    return t;
}

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
void test_linear_solver(std::size_t dim, double tolerance)
{
    dense_matrix<value_t> R(dim, dim);

    for (std::size_t i = 0; i < dim; ++i)
    {
        R(i, i) = 1.;
        for (std::size_t j = 0; j < i; ++j)
        {
            R(i, j) = static_cast<value_t>(0.4 * rand() / RAND_MAX);
            R(j, i) = 0.;
        }
    }

    std::vector<value_t>      x_ref(dim);
    std::vector<quarisma_int> pivot(dim + 1);

    for (std::size_t i = 0; i < dim; i++)
    {
        x_ref[i] = static_cast<value_t>(rand()) / static_cast<value_t>(RAND_MAX);
    }

    dense_matrix<value_t> A(dim, dim);
    {
        auto t_R = transpose(R);
        A        = matmul(R, t_R);
    }

    std::vector<value_t> b = matvec(A, x_ref);

    std::vector<value_t> x = b;
    linalg::linear_solver(
        R.begin(),
        pivot.data(),
        static_cast<quarisma_int>(dim),
        x.data(),
        linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);

    auto diff = hmax_abs_diff(x_ref, x);
    EXPECT_LE(diff, tolerance);

    x        = b;
    auto R2  = A;
    linalg::linear_solver(
        R2.begin(),
        pivot.data(),
        static_cast<quarisma_int>(dim),
        x.data(),
        linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);

    diff = hmax_abs_diff(x_ref, x);
    EXPECT_LE(diff, tolerance);

    x       = b;
    auto R3 = A;
    linalg::linear_solver(
        R3.begin(),
        pivot.data(),
        static_cast<quarisma_int>(dim),
        x.data(),
        linalg::linear_solver_type::LU_LINEAR_SOLVER);

    diff = hmax_abs_diff(x_ref, x);
    EXPECT_LE(diff, tolerance);
}
}  // namespace

TEST(Math, LinearSolver)
{
    const std::size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    test_linear_solver<float>(dim, 1.e-2);
    test_linear_solver<double>(dim, 1.e-10);
}
