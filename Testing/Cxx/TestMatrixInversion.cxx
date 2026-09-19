#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "linear_algebra/matrix_operation/matrix_inversion.h"

namespace
{
using linalg_test::dense_matrix;

template <typename>
struct Tolerance
{
};

template <>
struct Tolerance<float>
{
    static constexpr float value = 0.0002F;
};

template <>
struct Tolerance<double>
{
    static constexpr double value = 5.e-12;
};

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
double hmax_abs_diff(const dense_matrix<value_t>& a, const dense_matrix<value_t>& b)
{
    double max_error = 0;
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
void test_inversion(std::size_t dim)
{
    dense_matrix<value_t> R(dim, dim);
    dense_matrix<value_t> Id(dim, dim);

    for (std::size_t i = 0; i < dim; ++i)
    {
        Id(i, i) = 1.;
        R(i, i)  = 1.;
        for (std::size_t j = 0; j < i; ++j)
        {
            R(j, i) = 0;
            R(i, j) = static_cast<value_t>(0.4 * rand() / RAND_MAX);
        }
    }

    std::vector<quarisma_int> pivot(dim + 1);

    dense_matrix<value_t> A(dim, dim);
    {
        auto t_R = transpose(R);
        A        = matmul(R, t_R);
    }
    dense_matrix<value_t> IA = A;
    linalg::matrix_invert(
        IA.data(), pivot.data(), static_cast<quarisma_int>(dim), linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);

    for (std::size_t i = 0; i < dim; i++)
    {
        for (std::size_t j = 0; j < i; j++)
        {
            IA(j, i) = IA(i, j);
        }
    }

    R                 = matmul(IA, A);
    double max_error = hmax_abs_diff(R, Id);
    EXPECT_LE(max_error, Tolerance<value_t>::value);

    IA = A;
    linalg::matrix_invert(
        IA.data(), pivot.data(), static_cast<quarisma_int>(dim), linalg::linear_solver_type::LU_LINEAR_SOLVER);

    R         = matmul(IA, A);
    max_error = hmax_abs_diff(R, Id);
    EXPECT_LE(max_error, Tolerance<value_t>::value);

    linalg::matrix_determinant(
        A.data(), pivot.data(), static_cast<quarisma_int>(dim), linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);

    linalg::matrix_determinant(
        A.data(), pivot.data(), static_cast<quarisma_int>(dim), linalg::linear_solver_type::LU_LINEAR_SOLVER);

    EXPECT_ANY_THROW({
        linalg::matrix_invert(
            IA.data(), pivot.data(), static_cast<quarisma_int>(dim), (linalg::linear_solver_type)5);
    });

    EXPECT_ANY_THROW({
        linalg::matrix_determinant(
            IA.data(), pivot.data(), static_cast<quarisma_int>(dim), (linalg::linear_solver_type)5);
    });
}
}  // namespace

TEST(Math, MatrixInversion)
{
    const std::size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    test_inversion<double>(dim);
    test_inversion<float>(dim);

    test_inversion<double>(3);
    test_inversion<float>(3);
}
