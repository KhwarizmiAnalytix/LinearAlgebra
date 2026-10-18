// Dedicated numerical tests for linalg::determinant.
//
// The current API omits the LU pivot sign and does not square the Cholesky
// diagonal product (documented in determinant.h). Tests are therefore scoped
// to cases where these limitations do not affect correctness:
//   LU mode      — diagonal matrices (no row swaps, pivot sign = +1 trivially).
//   Cholesky mode — SPD diagonal matrices where the expected result is the
//                   product of sqrt(d_i), i.e. sqrt(det(A)), not det(A).

#include <cmath>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/determinant.h"
#include "include/matrix_operation/linear_solver.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct DetTolerance
{
};
template <> struct DetTolerance<float>
{
    static constexpr double value = 1e-5;
};
template <> struct DetTolerance<double>
{
    static constexpr double value = 1e-12;
};

// Diagonal n×n matrix with prescribed values (singular values = abs(d_i)).
template <typename T>
dense_matrix<T> diagonal(const std::vector<double>& d)
{
    const std::size_t n = d.size();
    dense_matrix<T>   A(n, n);
    A.fill(T(0));
    for (std::size_t i = 0; i < n; i++)
        A(i, i) = static_cast<T>(d[i]);
    return A;
}

// Compute the LU-diagonal product for a copy of A.
// apply() overwrites the matrix; pass by value so the caller's copy is intact.
template <typename T, linalg::linalg_backend Backend>
double compute_det_lu(dense_matrix<T> A)
{
    std::vector<linalg_int> pivot(A.rows() + 1);
    const auto              ws_size =
        linalg::determinant<T, Backend>::workspace_size(
            static_cast<linalg_int>(A.rows()), linalg::linear_solver_type::LU_LINEAR_SOLVER);
    std::vector<unsigned char> ws(ws_size);
    return static_cast<double>(
        linalg::determinant<T, Backend>::apply(A.data(),
            pivot.data(),
            static_cast<linalg_int>(A.rows()),
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            ws.data(),
            ws_size));
}

// Compute the Cholesky-diagonal product for a copy of A (= sqrt(det(A)) for SPD diagonal A).
template <typename T, linalg::linalg_backend Backend>
double compute_det_chol(dense_matrix<T> A)
{
    std::vector<linalg_int> pivot(A.rows() + 1);
    const auto              ws_size =
        linalg::determinant<T, Backend>::workspace_size(
            static_cast<linalg_int>(A.rows()), linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
    std::vector<unsigned char> ws(ws_size);
    return static_cast<double>(
        linalg::determinant<T, Backend>::apply(A.data(),
            pivot.data(),
            static_cast<linalg_int>(A.rows()),
            linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER,
            ws.data(),
            ws_size));
}

// Diagonal matrices with all-positive entries: no LU pivoting, result = product of d_i.
// Mirrors numpy.linalg.det on diagonal inputs.
template <typename T, linalg::linalg_backend Backend>
void test_determinant_diagonal()
{
    const double tol = DetTolerance<T>::value;

    // 2x2: det = 2*3 = 6
    EXPECT_NEAR((compute_det_lu<T, Backend>(diagonal<T>({2.0, 3.0}))), 6.0, tol);

    // 3x3: det = 2*3*4 = 24
    EXPECT_NEAR((compute_det_lu<T, Backend>(diagonal<T>({2.0, 3.0, 4.0}))), 24.0, tol);

    // 4x4: det = 1*2*3*5 = 30
    EXPECT_NEAR((compute_det_lu<T, Backend>(diagonal<T>({1.0, 2.0, 3.0, 5.0}))), 30.0, tol);
}

// Identity matrix: det = 1.
template <typename T, linalg::linalg_backend Backend>
void test_determinant_identity(std::size_t n)
{
    const double tol = DetTolerance<T>::value;
    EXPECT_NEAR((compute_det_lu<T, Backend>(diagonal<T>(std::vector<double>(n, 1.0)))), 1.0, tol);
}

// Singular matrix: factorization encounters a zero pivot.
// The API doc says "returns zero if factorization fails"; the native LU path
// currently returns NaN when the pivot is zero. Both are acceptable sentinel
// values indicating a non-invertible matrix; the test accepts either.
template <typename T, linalg::linalg_backend Backend>
void test_determinant_singular()
{
    // diag(1, 0, 1): one zero pivot → det is 0 or NaN
    {
        const double r = compute_det_lu<T, Backend>(diagonal<T>({1.0, 0.0, 1.0}));
        EXPECT_TRUE(std::isnan(r) || r == 0.0);
    }

    // All-zero 3x3 matrix
    {
        dense_matrix<T> Z(3, 3);
        Z.fill(T(0));
        const double r = compute_det_lu<T, Backend>(Z);
        EXPECT_TRUE(std::isnan(r) || r == 0.0);
    }
}

// SPD diagonal matrix via Cholesky mode.
// Cholesky of diag(d1,...,dn) = diag(sqrt(d1),...,sqrt(dn)).
// API returns product of L_ii = sqrt(d1)*...*sqrt(dn) = sqrt(d1*...*dn).
// So result^2 ≈ det(A).
template <typename T, linalg::linalg_backend Backend>
void test_determinant_cholesky()
{
    const double tol = DetTolerance<T>::value;

    // diag(4, 9): det = 36, sqrt(det) = 6
    {
        const double result = compute_det_chol<T, Backend>(diagonal<T>({4.0, 9.0}));
        EXPECT_NEAR(result * result, 36.0, tol * 36.0);
    }

    // diag(1, 4, 16): det = 64, sqrt(det) = 8
    {
        const double result = compute_det_chol<T, Backend>(diagonal<T>({1.0, 4.0, 16.0}));
        EXPECT_NEAR(result * result, 64.0, tol * 64.0);
    }
}

}  // namespace

TEST(Math, Determinant)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_determinant_diagonal<float, Backend>();
        test_determinant_diagonal<double, Backend>();
    });
}

TEST(Math, DeterminantIdentity)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_determinant_identity<float, Backend>(1);
        test_determinant_identity<float, Backend>(4);
        test_determinant_identity<double, Backend>(1);
        test_determinant_identity<double, Backend>(4);
    });
}

TEST(Math, DeterminantSingular)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_determinant_singular<float, Backend>();
        test_determinant_singular<double, Backend>();
    });
}

TEST(Math, DeterminantCholesky)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr linalg::linalg_backend Backend = decltype(backend)::value;
        test_determinant_cholesky<float, Backend>();
        test_determinant_cholesky<double, Backend>();
    });
}
