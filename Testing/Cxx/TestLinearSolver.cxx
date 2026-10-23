#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "test_backend.h"
#include "test_reporting.h"
#include "gtest/gtest.h"

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

template <typename value_t, linalg::linalg_backend Backend>
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

    std::vector<value_t>    x_ref(dim);
    std::vector<linalg_int> pivot(dim + 1);

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

    auto       lu_factors = A;
    const auto upfront_lu_ws_size =
        linalg::lu_decomposition<value_t, Backend>::workspace_size(static_cast<linalg_int>(dim));
    std::vector<unsigned char> upfront_lu_ws(upfront_lu_ws_size);
    const bool upfront_lu_ok = linalg::lu_decomposition<value_t, Backend>::apply(lu_factors.data(),
        static_cast<linalg_int>(dim),
        pivot.data(),
        upfront_lu_ws.data(),
        upfront_lu_ws_size);
    ASSERT_TRUE(upfront_lu_ok);
    auto lu_solution = b;
    linalg::linear_solver<value_t, Backend>::apply(lu_factors.data(),
        pivot.data(),
        static_cast<linalg_int>(dim),
        lu_solution.data(),
        linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER,
        nullptr,
        0);
    EXPECT_LE(hmax_abs_diff(x_ref, lu_solution), tolerance);

    std::vector<value_t> x = b;
    linalg::linear_solver<value_t, Backend>::apply(R.begin(),
        pivot.data(),
        static_cast<linalg_int>(dim),
        x.data(),
        linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER,
        nullptr,
        0);

    auto diff = hmax_abs_diff(x_ref, x);
    EXPECT_LE(diff, tolerance);

    x       = b;
    auto R2 = A;
    linalg::linear_solver<value_t, Backend>::apply(R2.begin(),
        pivot.data(),
        static_cast<linalg_int>(dim),
        x.data(),
        linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER,
        nullptr,
        0);

    diff = hmax_abs_diff(x_ref, x);
    EXPECT_LE(diff, tolerance);

    x                     = b;
    auto       R3         = A;
    const auto lu_ws_size = linalg::linear_solver<value_t, Backend>::workspace_size(
        static_cast<linalg_int>(dim), linalg::linear_solver_type::LU_LINEAR_SOLVER);
    std::vector<unsigned char> lu_ws(lu_ws_size);
    linalg::linear_solver<value_t, Backend>::apply(R3.begin(),
        pivot.data(),
        static_cast<linalg_int>(dim),
        x.data(),
        linalg::linear_solver_type::LU_LINEAR_SOLVER,
        lu_ws.data(),
        lu_ws_size);

    diff = hmax_abs_diff(x_ref, x);
    EXPECT_LE(diff, tolerance);
}
}  // namespace

TEST(Math, LinearSolver)
{
    const std::size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    linalg_test::for_each_available_backend(
        [&](auto backend)
        {
            constexpr linalg::linalg_backend Backend = decltype(backend)::value;
            test_linear_solver<float, Backend>(dim, 1.e-2);
            test_linear_solver<double, Backend>(dim, 1.e-10);
        });
}
