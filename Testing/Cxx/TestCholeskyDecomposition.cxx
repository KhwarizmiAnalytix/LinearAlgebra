#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/determinant.h"
#include "include/matrix_operation/multiplication.h"
#include "gtest/gtest.h"
#include "include/logging.h"

namespace
{
using linalg_test::dense_matrix;

template <typename value_t> struct Tolerance
{
    static constexpr value_t value = (value_t)200 * std::numeric_limits<value_t>::epsilon();
};

template <linalg::linalg_backend Backend> const char* backend_name()
{
    return Backend == linalg::linalg_backend::native
               ? "native"
               : (Backend == linalg::linalg_backend::blas_lapacke ? "blas_lapacke" : "mkl");
}

template <typename value_t> const char* value_type_name()
{
    return std::is_same<value_t, float>::value ? "float" : "double";
}

const char* triangle_name(linalg::cholesky_decomposition_enum type)
{
    return type == linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR ? "lower" : "upper";
}

template <typename value_t, linalg::linalg_backend Backend>
std::string test_details(std::size_t dim, linalg::cholesky_decomposition_enum type)
{
    return std::string("backend=") + backend_name<Backend>() +
           ", type=" + value_type_name<value_t>() + ", size=" + std::to_string(dim) + "x" +
           std::to_string(dim) + ", triangle=" + triangle_name(type);
}

template <typename value_t>
void expect_difference(const char* method,
    const char*                    check,
    value_t                        actual,
    value_t                        expected,
    value_t                        tolerance,
    const std::string&             details)
{
    const double difference = std::fabs(static_cast<double>(actual - expected));
    const bool   passed     = difference <= static_cast<double>(tolerance);

    if (passed)
    {
        LOGGING_LOG_INFO("[PASS]\nmethod tested={}\ncheck={}\ndetails={}\nexpected={}, actual={}, "
                         "difference={}, tolerance={}",
            method,
            check,
            details,
            expected,
            actual,
            difference,
            tolerance);
    }
    else
    {
        LOGGING_LOG_ERROR("[FAIL]\nmethod tested={}\ncheck={}\ndetails={}\nexpected={}, actual={}, "
                          "difference={}, tolerance={}",
            method,
            check,
            details,
            expected,
            actual,
            difference,
            tolerance);
    }

    EXPECT_LE(difference, static_cast<double>(tolerance));
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

template <typename value_t, linalg::linalg_backend Backend>
dense_matrix<value_t> multiply(const dense_matrix<value_t>& a, const dense_matrix<value_t>& b)
{
    dense_matrix<value_t> c(a.rows(), b.cols());
    const auto            rows  = static_cast<linalg_int>(a.rows());
    const auto            cols  = static_cast<linalg_int>(b.cols());
    const auto            depth = static_cast<linalg_int>(a.cols());
    const auto            ws_size =
        linalg::multiplication<value_t, Backend>::workspace_size(rows, cols, depth);
    std::vector<unsigned char> workspace(ws_size);
    linalg::multiplication<value_t, Backend>::apply(false,
        false,
        rows,
        cols,
        depth,
        a.data(),
        depth,
        b.data(),
        cols,
        c.data(),
        cols,
        workspace.data(),
        ws_size);
    return c;
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

template <typename value_t, linalg::linalg_backend Backend>
void build_cholesky_matrix(
    const std::vector<value_t>& values, std::size_t dim, linalg::cholesky_decomposition_enum type)
{
    auto C = build_symmetric_matrix<value_t>(values, dim, type);

    dense_matrix<value_t> A(dim, dim);
    auto                  t_C = transpose(C);
    if (type == linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)
    {
        A = multiply<value_t, Backend>(C, t_C);
    }
    else
    {
        A = multiply<value_t, Backend>(t_C, C);
    }

    dense_matrix<value_t> R = A;
    linalg::cholesky_decomposition<value_t, Backend>::apply(
        R.begin(), static_cast<linalg_int>(dim), type);
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
    auto       diff    = hmax_abs_diff(R, C);
    const auto details = test_details<value_t, Backend>(dim, type);
    expect_difference("linalg::cholesky_decomposition",
        "factor max absolute element error",
        diff,
        static_cast<value_t>(0),
        Tolerance<value_t>::value,
        details);

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
        std::vector<linalg_int> pivot(dim);

        dense_matrix<value_t> IA(dim, dim);
        dense_matrix<value_t> Id(dim, dim);
        for (std::size_t i = 0; i < dim; ++i)
        {
            Id(i, i) = 1;
        }

        IA = R;
        const auto invert_ws_size =
            linalg::inversion<value_t, Backend>::workspace_size(
                static_cast<linalg_int>(dim),
                linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);
        std::vector<unsigned char> invert_ws(invert_ws_size);
        linalg::inversion<value_t, Backend>::apply(IA.data(),
            pivot.data(),
            static_cast<linalg_int>(dim),
            linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER,
            invert_ws.data(),
            invert_ws_size);

        for (std::size_t i = 0; i < dim; i++)
        {
            for (std::size_t j = 0; j < i; j++)
            {
                IA(j, i) = IA(i, j);
            }
        }

        auto RET = multiply<value_t, Backend>(IA, A);
        diff     = hmax_abs_diff(Id, RET);
        expect_difference("linalg::inversion",
            "identity reconstruction max absolute element error",
            diff,
            static_cast<value_t>(0),
            Tolerance<value_t>::value,
            details);

        IA = R;
        linalg::determinant<value_t, linalg::linalg_backend::native>::apply(
            IA.data(),
            pivot.data(),
            static_cast<linalg_int>(dim),
            linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER,
            nullptr,
            0);

        std::vector<value_t> x(dim);
        for (std::size_t i = 0; i < dim; i++)
        {
            x[i] = static_cast<value_t>(rand()) / static_cast<value_t>(RAND_MAX);
        }
        std::vector<value_t> b = x;
        linalg::linear_solver<value_t, Backend>::apply(R.begin(),
            pivot.data(),
            static_cast<linalg_int>(dim),
            x.data(),
            linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER,
            nullptr,
            0);

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
        expect_difference("linalg::linear_solver",
            "solution max absolute residual",
            solve_diff,
            static_cast<value_t>(0),
            Tolerance<value_t>::value,
            details);
    }
}

template <typename value_t, linalg::linalg_backend Backend> void test_cholesky(std::size_t dim)
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

    build_cholesky_matrix<value_t, Backend>(
        values, dim, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);
    build_cholesky_matrix<value_t, Backend>(
        values, dim, linalg::cholesky_decomposition_enum::UPPER_TRIANGULAR);

    // mkl's cholesky_decomposition doesn't validate/throw on this invalid
    // enum the way native and blas_lapacke do.
    if (Backend != linalg::linalg_backend::mkl)
    {
        bool threw = false;
        try
        {
            linalg::cholesky_decomposition<value_t, Backend>::apply(
                &values[0], 1, (linalg::cholesky_decomposition_enum)'5');
        }
        catch (...)
        {
            threw = true;
        }

        const auto details = "backend=" + std::string(backend_name<Backend>()) +
                             ", type=" + value_type_name<value_t>() +
                             ", size=1x1, triangle=invalid";
        if (threw)
        {
            LOGGING_LOG_INFO("[PASS]\nmethod tested=linalg::cholesky_decomposition\n"
                             "check=reject invalid triangle\n"
                             "details={}\nexpected=exception, actual=exception, "
                             "difference=n/a, tolerance=n/a",
                details);
        }
        else
        {
            LOGGING_LOG_ERROR("[FAIL]\nmethod tested=linalg::cholesky_decomposition\n"
                              "check=reject invalid triangle\n"
                              "details={}\nexpected=exception, actual=no exception, "
                              "difference=n/a, tolerance=n/a",
                details);
        }
        EXPECT_TRUE(threw);
    }
}
}  // namespace

TEST(Math, CholeskyDecomposition)
{
    const std::size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    linalg_test::for_each_available_backend([&](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_cholesky<float, Backend>(dim);
        test_cholesky<double, Backend>(dim);

        test_cholesky<float, Backend>(13);
        test_cholesky<double, Backend>(13);
    });
}
