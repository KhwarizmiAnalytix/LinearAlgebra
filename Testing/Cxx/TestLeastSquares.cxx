#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/least_squares.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct LeastSquaresTolerance
{
};
template <> struct LeastSquaresTolerance<float>
{
    static constexpr double value = 1e-4;
};
template <> struct LeastSquaresTolerance<double>
{
    static constexpr double value = 1e-10;
};

// Overdetermined, consistent system (B = A * X_true exactly): the
// least-squares solution should recover X_true (residual is zero).
template <typename value_t, linalg::linalg_backend Backend>
void test_least_squares_overdetermined()
{
    const std::size_t rows = 6, columns = 3, nrhs = 2;

    dense_matrix<value_t> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            A(i, j) = static_cast<value_t>(1. + (i * 3 + j * 7) % 11);
        }
    }

    dense_matrix<value_t> X_true(columns, nrhs);
    for (std::size_t i = 0; i < columns; i++)
    {
        for (std::size_t j = 0; j < nrhs; j++)
        {
            X_true(i, j) = static_cast<value_t>(1 + i + 2 * j);
        }
    }

    const auto B = linalg_test::matmul(A, X_true);

    dense_matrix<value_t> X(columns, nrhs);
    const auto            ws_size =
        linalg::least_squares<value_t, Backend>::workspace_size(
            static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> ws(ws_size);
    linalg::least_squares<value_t, Backend>::apply(
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(nrhs),
        A.begin(),
        static_cast<linalg_long>(columns),
        B.begin(),
        static_cast<linalg_long>(nrhs),
        X.begin(),
        static_cast<linalg_long>(nrhs),
        ws.data(),
        ws_size);

    value_t max_error = 0;
    for (std::size_t i = 0; i < columns; i++)
    {
        for (std::size_t j = 0; j < nrhs; j++)
        {
            max_error =
                std::max(max_error, static_cast<value_t>(std::fabs(X(i, j) - X_true(i, j))));
        }
    }
    EXPECT_LT(max_error, static_cast<value_t>(LeastSquaresTolerance<value_t>::value));
}

// Underdetermined system: any solution satisfying A*X = B is a valid
// least-squares (zero-residual) answer; least_squares_solve should at
// least satisfy the system, since it is exactly solvable when rows < columns
// and A has full row rank.
template <typename value_t, linalg::linalg_backend Backend>
void test_least_squares_underdetermined()
{
    const std::size_t rows = 3, columns = 6, nrhs = 1;

    dense_matrix<value_t> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            A(i, j) = static_cast<value_t>(1. + (i * 5 + j * 2) % 7);
        }
    }

    dense_matrix<value_t> B(rows, nrhs);
    for (std::size_t i = 0; i < rows; i++)
    {
        B(i, 0) = static_cast<value_t>(3 + i);
    }

    dense_matrix<value_t> X(columns, nrhs);
    const auto            ws_size =
        linalg::least_squares<value_t, Backend>::workspace_size(
            static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> ws(ws_size);
    linalg::least_squares<value_t, Backend>::apply(
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(nrhs),
        A.begin(),
        static_cast<linalg_long>(columns),
        B.begin(),
        static_cast<linalg_long>(nrhs),
        X.begin(),
        static_cast<linalg_long>(nrhs),
        ws.data(),
        ws_size);

    value_t max_error = 0;
    for (std::size_t i = 0; i < rows; i++)
    {
        value_t sum = 0;
        for (std::size_t k = 0; k < columns; k++)
        {
            sum += A(i, k) * X(k, 0);
        }
        max_error = std::max(max_error, static_cast<value_t>(std::fabs(sum - B(i, 0))));
    }
    EXPECT_LT(max_error, static_cast<value_t>(LeastSquaresTolerance<value_t>::value));
}

// Square, full-rank consistent system: rows == columns, exact solution exists.
// Mirrors torch.linalg.lstsq behavior on square inputs (PyTorch test_lstsq).
template <typename value_t, linalg::linalg_backend Backend>
void test_least_squares_square()
{
    const std::size_t n = 4, nrhs = 2;

    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            A(i, j) = static_cast<value_t>((i == j) ? 5.0 + i : 0.5 * ((i + j + 1) % 3));
        }
    }

    dense_matrix<value_t> X_true(n, nrhs);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < nrhs; j++)
        {
            X_true(i, j) = static_cast<value_t>(1 + i - j);
        }
    }

    const auto B = linalg_test::matmul(A, X_true);

    dense_matrix<value_t> X(n, nrhs);
    const auto            ws_size =
        linalg::least_squares<value_t, Backend>::workspace_size(
            static_cast<linalg_long>(n), static_cast<linalg_long>(n));
    std::vector<unsigned char> ws(ws_size);
    linalg::least_squares<value_t, Backend>::apply(
        static_cast<linalg_long>(n),
        static_cast<linalg_long>(n),
        static_cast<linalg_long>(nrhs),
        A.begin(),
        static_cast<linalg_long>(n),
        B.begin(),
        static_cast<linalg_long>(nrhs),
        X.begin(),
        static_cast<linalg_long>(nrhs),
        ws.data(),
        ws_size);

    value_t max_error = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < nrhs; j++)
        {
            max_error =
                std::max(max_error, static_cast<value_t>(std::fabs(X(i, j) - X_true(i, j))));
        }
    }
    EXPECT_LT(max_error, static_cast<value_t>(LeastSquaresTolerance<value_t>::value));
}

// Rank-deficient overdetermined system: A is a rank-1 outer product (u*v^T) so
// the column space is a line. The solution must still minimise ||A*X - B||_F.
// Mirrors torch.linalg.lstsq on rank-deficient inputs (PyTorch test_lstsq_rank_deficient).
template <typename value_t, linalg::linalg_backend Backend>
void test_least_squares_rank_deficient()
{
    const std::size_t rows = 6, columns = 3, nrhs = 1;

    // Build a rank-1 matrix A = u * v^T.
    std::vector<value_t> u(rows), v(columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        u[i] = static_cast<value_t>(i + 1);
    }
    for (std::size_t j = 0; j < columns; j++)
    {
        v[j] = static_cast<value_t>(j + 1);
    }

    dense_matrix<value_t> A(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            A(i, j) = u[i] * v[j];
        }
    }

    // B lives in the column space of A, so a zero-residual solution exists.
    dense_matrix<value_t> B(rows, nrhs);
    for (std::size_t i = 0; i < rows; i++)
    {
        B(i, 0) = u[i] * static_cast<value_t>(2);
    }

    dense_matrix<value_t> X(columns, nrhs);
    const auto            ws_size =
        linalg::least_squares<value_t, Backend>::workspace_size(
            static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> ws(ws_size);
    linalg::least_squares<value_t, Backend>::apply(
        static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        static_cast<linalg_long>(nrhs),
        A.begin(),
        static_cast<linalg_long>(columns),
        B.begin(),
        static_cast<linalg_long>(nrhs),
        X.begin(),
        static_cast<linalg_long>(nrhs),
        ws.data(),
        ws_size);

    // Residual ||A*X - B|| should be near zero since B is in range(A).
    value_t max_residual = 0;
    for (std::size_t i = 0; i < rows; i++)
    {
        value_t sum = 0;
        for (std::size_t k = 0; k < columns; k++)
        {
            sum += A(i, k) * X(k, 0);
        }
        max_residual = std::max(max_residual, static_cast<value_t>(std::fabs(sum - B(i, 0))));
    }
    EXPECT_LT(max_residual, static_cast<value_t>(LeastSquaresTolerance<value_t>::value));
}

// Trivial 1x1 system: scalar division. Mirrors PyTorch edge-case coverage.
template <typename value_t, linalg::linalg_backend Backend>
void test_least_squares_1x1()
{
    const value_t             a_val = static_cast<value_t>(3);
    const value_t             b_val = static_cast<value_t>(12);
    dense_matrix<value_t>     A(1, 1);
    dense_matrix<value_t>     B(1, 1);
    dense_matrix<value_t>     X(1, 1);
    A(0, 0)        = a_val;
    B(0, 0)        = b_val;
    const auto ws_size =
        linalg::least_squares<value_t, Backend>::workspace_size(1, 1);
    std::vector<unsigned char> ws(ws_size);
    linalg::least_squares<value_t, Backend>::apply(
        1, 1, 1, A.begin(), 1, B.begin(), 1, X.begin(), 1, ws.data(), ws_size);
    EXPECT_NEAR(static_cast<double>(X(0, 0)), static_cast<double>(b_val / a_val), LeastSquaresTolerance<value_t>::value);
}
}  // namespace

TEST(Math, LeastSquares)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_least_squares_overdetermined<float, Backend>();
        test_least_squares_overdetermined<double, Backend>();
        test_least_squares_underdetermined<float, Backend>();
        test_least_squares_underdetermined<double, Backend>();
    });
}

TEST(Math, LeastSquaresSquare)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_least_squares_square<float, Backend>();
        test_least_squares_square<double, Backend>();
    });
}

TEST(Math, LeastSquaresRankDeficient)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_least_squares_rank_deficient<float, Backend>();
        test_least_squares_rank_deficient<double, Backend>();
    });
}

TEST(Math, LeastSquares1x1)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_least_squares_1x1<float, Backend>();
        test_least_squares_1x1<double, Backend>();
    });
}
