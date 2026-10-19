#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/determinant.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct Tolerance
{
};

template <> struct Tolerance<float>
{
    static constexpr float value = 0.0002F;
};

template <> struct Tolerance<double>
{
    static constexpr double value = 5.e-12;
};

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

template <typename value_t, linalg::linalg_backend Backend> void test_inversion(std::size_t dim)
{
    SCOPED_TRACE(::testing::Message() << "backend=" << static_cast<int>(Backend)
                                   << ", scalar_bytes=" << sizeof(value_t) << ", dimension=" << dim);
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

    std::vector<linalg_int> pivot(dim + 1);

    dense_matrix<value_t> A(dim, dim);
    {
        auto t_R = transpose(R);
        A        = matmul(R, t_R);
    }
    dense_matrix<value_t> IA = A;
    {
        const auto ws_size =
            linalg::inversion<value_t, Backend>::workspace_size(
                static_cast<linalg_int>(dim), linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
        std::vector<unsigned char> ws(ws_size);
        linalg::inversion<value_t, Backend>::apply(IA.data(),
            pivot.data(),
            static_cast<linalg_int>(dim),
            linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER,
            ws.data(),
            ws_size);
    }

    for (std::size_t i = 0; i < dim; i++)
    {
        for (std::size_t j = 0; j < i; j++)
        {
            IA(j, i) = IA(i, j);
        }
    }

    R                = matmul(IA, A);
    double max_error = hmax_abs_diff(R, Id);
    EXPECT_LE(max_error, Tolerance<value_t>::value);

    IA = A;
    {
        const auto ws_size =
            linalg::inversion<value_t, Backend>::workspace_size(
                static_cast<linalg_int>(dim), linalg::linear_solver_type::LU_LINEAR_SOLVER);
        std::vector<unsigned char> ws(ws_size);
        linalg::inversion<value_t, Backend>::apply(IA.data(),
            pivot.data(),
            static_cast<linalg_int>(dim),
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            ws.data(),
            ws_size);
    }

    R         = matmul(IA, A);
    max_error = hmax_abs_diff(R, Id);
    EXPECT_LE(max_error, Tolerance<value_t>::value);

    {
        const auto                 ws_size = linalg::determinant<value_t,
                            linalg::linalg_backend::native>::workspace_size(static_cast<linalg_int>(dim),
            linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
        std::vector<unsigned char> ws(ws_size);
        linalg::determinant<value_t, linalg::linalg_backend::native>::apply(A.data(),
            pivot.data(),
            static_cast<linalg_int>(dim),
            linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER,
            ws.data(),
            ws_size);
    }

    {
        const auto                 ws_size = linalg::determinant<value_t,
                            linalg::linalg_backend::native>::workspace_size(static_cast<linalg_int>(dim),
            linalg::linear_solver_type::LU_LINEAR_SOLVER);
        std::vector<unsigned char> ws(ws_size);
        linalg::determinant<value_t, linalg::linalg_backend::native>::apply(A.data(),
            pivot.data(),
            static_cast<linalg_int>(dim),
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            ws.data(),
            ws_size);
    }

    EXPECT_ANY_THROW((linalg::inversion<value_t, Backend>::apply(IA.data(),
        pivot.data(),
        static_cast<linalg_int>(dim),
        (linalg::linear_solver_type)5,
        nullptr,
        0)));

    EXPECT_ANY_THROW(
        (linalg::determinant<value_t, linalg::linalg_backend::native>::apply(
            IA.data(),
            pivot.data(),
            static_cast<linalg_int>(dim),
            (linalg::linear_solver_type)5,
            nullptr,
            0)));
}
}  // namespace

TEST(Math, MatrixInversion)
{
    const std::size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    linalg_test::for_each_available_backend([&](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_inversion<double, Backend>(dim);
        test_inversion<float, Backend>(dim);

        test_inversion<double, Backend>(3);
        test_inversion<float, Backend>(3);
    });
}
