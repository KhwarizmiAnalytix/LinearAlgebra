#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/matrix_inversion.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t>
struct Tolerance
{
    static constexpr value_t value = (value_t)200 * std::numeric_limits<value_t>::epsilon();
};

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

// Builds a triangular matrix (values placed either strictly-lower or
// strictly-upper depending on `type`) from a packed value list, matching the
// original test's construction.
template <typename value_t>
dense_matrix<value_t> build_symmetric_matrix(
    const std::vector<value_t>& values, std::size_t dim, linalg::cholesky_decomposition_enum type)
{
    dense_matrix<value_t> M(dim, dim);
    std::size_t           offset = 0;
    if (type == linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)
    {
        for (std::size_t i = 0; i < dim; ++i)
        {
            M(i, i) = values[offset++];
            for (std::size_t j = 0; j < i; ++j)
            {
                M(i, j) = values[offset++];
                M(j, i) = 0;
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < dim; ++i)
        {
            M(i, i) = values[offset++];
            for (std::size_t j = 0; j < i; ++j)
            {
                M(j, i) = values[offset++];
                M(i, j) = 0;
            }
        }
    }
    return M;
}

template <typename value_t>
void build_cholesky_matrix(
    const std::vector<value_t>& values, std::size_t dim, linalg::cholesky_decomposition_enum type)
{
    auto C = build_symmetric_matrix<value_t>(values, dim, type);

    dense_matrix<value_t> A(dim, dim);
    auto                  t_C = transpose(C);
    if (type == linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)
    {
        A = matmul(C, t_C);
    }
    else
    {
        A = matmul(t_C, C);
    }

    dense_matrix<value_t> R = A;
    linalg::cholesky_decomposition(R.begin(), static_cast<quarisma_int>(dim), type);
    for (std::size_t i = 0; i < dim; ++i)
    {
        for (std::size_t j = i + 1; j < dim; ++j)
        {
            if (type == linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)
            {
                R(i, j) = 0;
            }
            else
            {
                R(j, i) = 0;
            }
        }
    }
    auto diff = hmax_abs_diff(R, C);
    EXPECT_LE(diff, Tolerance<value_t>::value);

    // Zero out the opposite triangle so R is purely the Cholesky factor
    // before using it in matrix_invert / linear_solver below.
    if (type == linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)
    {
        for (std::size_t i = 0; i < dim; ++i)
        {
            for (std::size_t j = 0; j < i; ++j)
            {
                R(j, i) = 0;
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < dim; ++i)
        {
            for (std::size_t j = 0; j < i; ++j)
            {
                R(i, j) = 0;
            }
        }
    }

    if (type == linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)
    {
        std::vector<quarisma_int> pivot(dim);

        dense_matrix<value_t> IA(dim, dim);
        dense_matrix<value_t> Id(dim, dim);
        for (std::size_t i = 0; i < dim; ++i)
        {
            Id(i, i) = 1;
        }

        IA = R;
        linalg::matrix_invert(
            IA.data(),
            pivot.data(),
            static_cast<quarisma_int>(dim),
            linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);

        for (std::size_t i = 0; i < dim; i++)
        {
            for (std::size_t j = 0; j < i; j++)
            {
                IA(j, i) = IA(i, j);
            }
        }

        auto RET  = matmul(IA, A);
        diff      = hmax_abs_diff(Id, RET);
        EXPECT_LE(diff, Tolerance<value_t>::value);

        IA = R;
        linalg::matrix_determinant(
            IA.data(),
            pivot.data(),
            static_cast<quarisma_int>(dim),
            linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);

        std::vector<value_t> x(dim);
        for (std::size_t i = 0; i < dim; i++)
        {
            x[i] = static_cast<value_t>(rand()) / static_cast<value_t>(RAND_MAX);
        }
        std::vector<value_t> b = x;
        linalg::linear_solver(
            R.begin(),
            pivot.data(),
            static_cast<quarisma_int>(dim),
            x.data(),
            linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);

        // b - A * x
        value_t solve_diff = 0;
        for (std::size_t i = 0; i < dim; ++i)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < dim; ++k)
            {
                sum += A(i, k) * x[k];
            }
            solve_diff = std::fmax(solve_diff, std::fabs(static_cast<double>(b[i] - sum)));
        }
        EXPECT_LE(solve_diff, Tolerance<value_t>::value);
    }
}

template <typename value_t>
void test_cholesky(std::size_t dim)
{
    std::vector<value_t> values(dim * (dim + 1) / 2);

    std::size_t offset = 0;
    for (std::size_t i = 0; i < dim; ++i)
    {
        values[offset++] = static_cast<value_t>(5.);
        for (std::size_t j = 0; j < i; ++j)
        {
            values[offset++] = static_cast<value_t>(0.5 * rand() / RAND_MAX);
        }
    }

    build_cholesky_matrix<value_t>(values, dim, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);
    build_cholesky_matrix<value_t>(values, dim, linalg::cholesky_decomposition_enum::UPPER_TRIANGULAR);

#ifndef LINALG_ENABLE_MKL
    EXPECT_ANY_THROW({
        linalg::cholesky_decomposition(&values[0], 1, (linalg::cholesky_decomposition_enum)'5');
    });
#endif
}
}  // namespace

TEST(Math, CholeskyDecomposition)
{
    const std::size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    test_cholesky<float>(dim);
    test_cholesky<double>(dim);

    test_cholesky<float>(13);
    test_cholesky<double>(13);
}
